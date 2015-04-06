#ifndef _SORT_SIMD_H
#define _SORT_SIMD_H

#include <pmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <stdint.h>

/**
 * Ordena un registro __m128 de menor a mayor con una red simple de 
 * ordenamiento.
 *
 * @param reg Es el registro que se quiere ordenar
 */
__m128 sort_m128(__m128 reg) __attribute__((const)); 

/**
 * Invierte un registro.
 *
 * [R0 R1 R2 R3] --> [R3 R2 R1 R0]
 *
 * @param reg	Registro que se quiere invertir
 *
 * @return El registro a invertir
 */
inline __m128 reverse_m128(__m128 reg) __attribute__((const));

/**
 * Hace un mergesort entre de los registros 'a' y 'b' y los coloca en dest.  
 * 'dest' tiene que ser un arreglo de largo 2 de tipo «__m128».
 *
 * @param dest	Arreglo donde se colocan las 2 secuencias ordenadas de menor a 
 * mayor
 * @param a		Primer registro __m128
 * @param b		Segundo registro __m128
 */
void bitonic_merge_network(__m128 *dest, __m128 a, __m128 b) __attribute__((nonnull(1)));

/**
 * Segunda etapa de la red de ordenamiento bitonica. Esta etapa de la red se 
 * puede hacer aparte, pues los registros resultantes dependen unicamente del 
 * mismo registro de entrada, a diferencia de las etapas restantes que dependen 
 * de los resultados del segundo registro.
 *
 * @param reg	Registro __m128
 * @return El registro resultante de la segunda etapa de la BMN
 */
static __m128 bmn_2ndstage(__m128 reg) __attribute__((const));
#endif
