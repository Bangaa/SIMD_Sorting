#ifndef _SORT_SIMD_H
#define _SORT_SIMD_H

#include <pmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <stdint.h>

#define _MM_REVERSE_PS(reg) _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3,1,2,0))

/**
 * Ordena un registro __m128 de menor a mayor con una red simple de 
 * ordenamiento.
 *
 * @param reg Es el registro que se quiere ordenar
 */
__m128 sort_m128(__m128 reg);

/**
 * Invierte un registro.
 *
 * [R0 R1 R2 R3] --> [R3 R2 R1 R0]
 *
 * @param reg	Registro que se quiere invertir
 *
 * @return El registro a invertir
 */
//inline __m128 reverse_m128(__m128 reg) __attribute__((const));

/**
 * Red de Batcher. Hace un mergesort entre de los registros 'a' y 'b' y los 
 * coloca en dest. 'dest' tiene que ser un arreglo de largo 2 de tipo «__m128».
 *
 * @param dest	Arreglo donde se colocan las 2 secuencias ordenadas de menor a 
 * mayor
 * @param a		Primer registro __m128
 * @param b		Segundo registro __m128
 */
void bitonic_merge_network(__m128 *a, __m128 *b);

void in_register_sort(__m128 *a, __m128 *b, __m128 *c, __m128 *d);
#endif
