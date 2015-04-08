#ifndef __SORT_SIMD_H__
#define __SORT_SIMD_H__

#include <pmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <stdint.h>

#define _MM_INVERT_PS(reg) _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(0,1,2,3))

/**
 * Ordena un registro __m128 de menor a mayor con una red simple de 
 * ordenamiento.
 *
 * @param reg Es el registro que se quiere ordenar
 */
__m128 sort_m128(__m128 reg);

/**
 * Red de Batcher. Hace un mersort de una secuencia bitonica de 8 elementos 
 * separados en 2 registros. El primer registro (a) tiene que estar ordenado 
 * crecientemente y el segundo registro (b) tiene que estar ordenado 
 * decrecientemente.
 *
 * @param a		Primer registro __m128 monotonicamente creciente
 * @param b		Segundo registro __m128 monotonicamente decreciente
 */
void bitonic_merge_network(__m128 *a, __m128 *b);

/**
 * Toma 4 registros desordenados y entrega 4 registros ordenados de menor a 
 * mayor.
 *
 * @param a
 * @param b
 * @param c
 * @param d
 */
void in_register_sort(__m128 *a, __m128 *b, __m128 *c, __m128 *d);


/**
 * Mezcla 2 secuencias de 8 elementos, monotonicamente crecientes, dejando una 
 * secuencia de 16 elementos.  Toma 4 registros tales que «a» y «b» formen una 
 * secuencia de 8 valores ordenados de menor a mayor, asi como tambien «c» y 
 * «d». Al final mezcla las 2 secuencias (formadas por los registros de 
 * entrada) dejando una sola de 16 elementos ordenados de menor a mayor.
 *
 * @param a
 * @param b
 * @param c
 * @param d
 */
void merge_SIMD(__m128 *a, __m128 *b, __m128 *c, __m128 *d);
#endif
