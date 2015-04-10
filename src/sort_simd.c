#include "main.h"
#include "sort_simd.h"
#include <pmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <stdint.h>

/**
 * Segunda etapa de la red de ordenamiento bitonica. Esta etapa de la red se 
 * puede hacer aparte, pues los registros resultantes dependen unicamente del 
 * mismo registro de entrada, a diferencia de las etapas restantes que dependen 
 * de los resultados del segundo registro.
 *
 * @param reg	Registro __m128
 * @return El registro resultante de la segunda etapa de la BMN
 */
static __m128 bmn_2ndstage(__m128 reg);

__m128 sort_m128(__m128 reg)
{
	__m128 init, shuffled, min, max, aux;

	// etapa 1

	init = reg;
	shuffled = _mm_shuffle_ps(init, init, _MM_SHUFFLE(1,0,3,2));

	min = _mm_min_ps(init, shuffled);
	max = _mm_max_ps(init, shuffled);

	// etapa 2
	init = _mm_shuffle_ps(min, max, _MM_SHUFFLE(1,0,1,0));
	shuffled = _mm_shuffle_ps(init, init, _MM_SHUFFLE(2,3,0,1));

	min = _mm_min_ps(init, shuffled);
	max = _mm_max_ps(init, shuffled); 

	// etapa 3

	init = _mm_shuffle_ps(min, max, _MM_SHUFFLE(2,0,2,0));
	shuffled = _mm_shuffle_ps(init, init, _MM_SHUFFLE(3,1,2,0));

	min = _mm_min_ps(init, shuffled);
	max = _mm_max_ps(init, shuffled); 

	aux = _mm_shuffle_ps(min, max, _MM_SHUFFLE(1,1,1,1));
	aux = _mm_shuffle_ps(shuffled, aux, _MM_SHUFFLE(0,2,3,0));

	// registro ordenado:
	aux = _mm_shuffle_ps(aux, aux, _MM_SHUFFLE(1,2,3,0));

	return aux;
}

void bitonic_merge_network(__m128 *a, __m128 *b)
{
	__m128 reg_a, reg_b, min, max;

	reg_a = *a;
	reg_b = *b;

	min = _mm_min_ps(reg_a,reg_b);
	max = _mm_max_ps(reg_a,reg_b);

	reg_a = _mm_shuffle_ps(min, max, _MM_SHUFFLE(2,0,2,0));
	reg_b = _mm_shuffle_ps(min, max, _MM_SHUFFLE(3,1,3,1));

	reg_a = _mm_shuffle_ps(reg_a, reg_a, _MM_SHUFFLE(3,1,2,0));
	reg_b = _mm_shuffle_ps(reg_b, reg_b, _MM_SHUFFLE(3,1,2,0));

	// etapa 2 

	reg_a = bmn_2ndstage(reg_a);
	reg_b = bmn_2ndstage(reg_b); 

	// etapa 3 
	
	min = _mm_min_ps(reg_a,reg_b);
	max = _mm_max_ps(reg_a,reg_b);

	reg_a = _mm_shuffle_ps(min, max, _MM_SHUFFLE(1,0,1,0));
	reg_b = _mm_shuffle_ps(min, max, _MM_SHUFFLE(3,2,3,2));

	reg_a = _mm_shuffle_ps(reg_a, reg_a, _MM_SHUFFLE(3,1,2,0));
	reg_b = _mm_shuffle_ps(reg_b, reg_b, _MM_SHUFFLE(3,1,2,0));

	// se cargan registros al destino

	*a = reg_a;
	*b = reg_b;
}

static __m128 bmn_2ndstage(__m128 reg)
{
	__m128 reg_sh, min, max;

	reg_sh = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(1,0,3,2));

	min = _mm_min_ps(reg, reg_sh);
	max = _mm_max_ps(reg, reg_sh);

	reg = _mm_shuffle_ps(min, max, _MM_SHUFFLE(1,0,1,0));
	reg = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3,1,2,0));

	return reg;
}

void in_register_sort(__m128 *a, __m128 *b, __m128 *c, __m128 *d)
{
	__m128 ra, rb, rc, rd, min, max, min2, max2;

	ra = *a;
	rb = *b;
	rc = *c;
	rd = *d;

	min = _mm_min_ps(ra, rc);
	max = _mm_max_ps(ra, rc);

	min2 = _mm_min_ps(rb, rd);
	max2 = _mm_max_ps(rb, rd);

	ra = min;
	rb = min2;
	rc = max;
	rd = max2;

	// etapa 2

	min = _mm_min_ps(ra, rb);
	max = _mm_max_ps(ra, rb);

	min2 = _mm_min_ps(rc, rd);
	max2 = _mm_max_ps(rc, rd);

	ra = min;
	rb = max;
	rc = min2;
	rd = max2;

	// etapa 3

	min = _mm_min_ps(rb, rc);
	max = _mm_max_ps(rb, rc);

	rb = min;
	rc = max;

	_MM_TRANSPOSE4_PS(ra, rb, rc, rd);

	*a = ra;
	*b = rb;
	*c = rc;
	*d = rd;
}

void merge_SIMD(__m128 *a, __m128 *b, __m128 *c, __m128 *d)
{
	__m128 s1, s2, s3, s4;
	s1 = *a;
	s2 = *b;
	s3 = *c;
	s4 = *d;

	s3 = _MM_INVERT_PS(s3);
	bitonic_merge_network(&s1, &s3);

	*a = s1;

	// si s2[31:0] <= s4[31:0]
	if (_mm_ucomile_ss(s2, s4))
	{
		s3 = _MM_INVERT_PS(s3);
		bitonic_merge_network(&s2, &s3);
		*b = s2;

		s4 = _MM_INVERT_PS(s4);
		bitonic_merge_network(&s3, &s4);
		*c = s3;
		*d = s4; 
	}
	else
	{
		s4 = _MM_INVERT_PS(s4);
		bitonic_merge_network(&s3, &s4);
		*b = s3; 

		s4 = _MM_INVERT_PS(s4);
		bitonic_merge_network(&s2, &s4);
		*c = s2;
		*d = s4; 
	}
}

void sort_SIMD(__m128 *r1, __m128 *r2, __m128 *r3, __m128 *r4)
{
	// in-register sorting
	in_register_sort(r1, r2, r3, r4);

	// generar 2 secuencias ordenadas de a 8
	*r2 = _MM_INVERT_PS(*r2);
	*r4 = _MM_INVERT_PS(*r4);
	bitonic_merge_network(r1, r2);
	bitonic_merge_network(r3, r4);

	// generar secuencia de 16
	merge_SIMD(r1, r2, r3, r4);
}

void mw_merge_sort(float *dst, float *src, size_t nelm)
{
	int *ind; //<! indice de cada una de las listas
	size_t nlists = nelm / 16;
	int idx_dst = 0;

	ind = (int*) malloc(sizeof(int) * nlists);

	if (ind == NULL)
	{
		print_err("Error: no se pudo reservar memoria");
		exit(EXIT_FAILURE);
	}
	else
	{
		for (int i=0; i< nlists; i++)
			ind[i] = 0;
	}

	int j=0;  //<! Numero de la lista
	do
	{ 
		float min;
		float cand;
		int idx, idx_menor = j;
		float *lst_act;


		idx = ind[j];
		lst_act = src + 16*j;
		min = lst_act[idx];

		for (int i=j+1; i < nlists; i++)
		{
			idx = ind[i];
			if (idx > 15) continue;

			lst_act = src + 16*i;
			cand = lst_act[idx];

			if (cand < min)
			{
				min = cand;
				idx_menor = i;	//<! numero de la lista donde se encontro al menor
			}
		}
		dst[idx_dst++] = min;
		ind[idx_menor] = ind[idx_menor] + 1;

		while (ind[j] > 15)
			j++;

	}while(j < nlists);
}

void mw_merge_sortv2(float *array, size_t nelm)
{
	merge_sort(array, 0, nelm - 1);
}

void merge_sort(float *array, int inicio, int fin)
{
	int medio = (inicio + fin) >> 1;

	if ((fin - inicio) > 16)
	{ 
		merge_sort(array, inicio, medio);
		merge_sort(array, medio +1, fin);
		combinar(array, inicio, medio, fin);
	}
}

void combinar(float *array, int inicio, int medio, int fin)
{
    float *aux = (float*) calloc(fin-inicio+1, sizeof(float));
	int indAux, indFst, indSnd;
	int i; 
	indAux = 0;         //<! indice del arreglo auxiliar
	indFst = inicio;    //<! indice de la primera mitad
	indSnd = medio+1;   //<! indice de la segunda mitad
	
	while (indFst <= medio && indSnd <= fin)
	{
		//  cambiar <= por >= si se quiere ordenar de mayor a menor
		if (array[indFst] <= array[indSnd])
		{
			aux[indAux++] = array[indFst++];
		}
		else
		{
			aux[indAux++] = array[indSnd++];
		}
	}

	// Se copian los elementos de la primera mitad no comparados

	while (indFst<=medio)
	{
		aux[indAux++] = array[indFst++];
	}

	// Se copian los elementos de la segunda mitad no comparados

	while (indSnd <= fin)
	{
		aux[indAux++] = array[indSnd++];
	}

	indAux = 0;

	/* Finalmente se copian los elementos del array auxiliar (ordenados)
	   en el array original */

	for (i = inicio; i <= fin; i++)
	{
		array[i] = aux[indAux++];
	}
	free(aux);
}
