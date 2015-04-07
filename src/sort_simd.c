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
