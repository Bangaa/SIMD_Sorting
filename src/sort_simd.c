#include "sort_simd.h"
#include <pmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <stdint.h>


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

inline __m128 reverse_m128(__m128 reg)
{
	return _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(0,1,2,3));
}

void bitonic_merge_network(__m128 *dest, __m128 a, __m128 b)
{
	__m128 min, max;

	min = _mm_min_ps(a,b);
	max = _mm_max_ps(a,b);

	a = _mm_shuffle_ps(min, max, _MM_SHUFFLE(2,0,2,0));
	b = _mm_shuffle_ps(min, max, _MM_SHUFFLE(3,1,3,1));

	a = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3,1,2,0));
	b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3,1,2,0));

	// etapa 2 

	a = bmn_2ndstage(a);
	b = bmn_2ndstage(b); 

	// etapa 3 
	
	min = _mm_min_ps(a,b);
	max = _mm_max_ps(a,b);

	a = _mm_shuffle_ps(min, max, _MM_SHUFFLE(1,0,1,0));
	b = _mm_shuffle_ps(min, max, _MM_SHUFFLE(3,2,3,2));

	a = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3,1,2,0));
	b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3,1,2,0));

	// se cargan registros al destino

	dest[0] = a;
	dest[1] = b;
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
