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
 * Esta funcion no se hace uso en el programa pues hay una mejor forma de 
 * ordenar registros.
 *
 * @param reg Es el registro que se quiere ordenar
 */
__m128 sort_m128(__m128 reg) __attribute__((deprecated));

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

/**
 * Ordena 4 registros __m128, representando 16 valores, de menor a mayor. El 
 * resultado se guarda en los mismos registros de entrada.
 *
 * @param r1
 * @param r2
 * @param r3
 * @param r4
 */
void sort_SIMD(__m128 *r1, __m128 *r2, __m128 *r3, __m128 *r4);

/**
 * Junta todas las secuencias ordenadas de 16 elementos en una sola.
 * 
 * La funcion quedó obsoleta debido a que se implementa una mejor basada en el 
 * algoritmo merge-sort.
 *
 * @param dst	Arreglo de destino
 * @param src	Arreglo desde donde se sacan las secuencias
 */
void mw_merge_sort(float *dst, float *src, size_t nelm) __attribute__((deprecated));

/**
 * Funcion que hace la union entre las secuencias ordenadas de 16 elementos y 
 * las ordena dentro del mismo arreglo de entrada.
 *
 * @param array	Arreglo que se quiere ordenar
 * @param nelm	Largo del arreglo
 */
void mw_merge_sortv2(float *array, size_t nelm);

/**
 * Funcion recursiva que separa las secuencias de 2 en 2 para luego mezclarlas 
 * en una sola.
 *
 * @param array   Arreglo con las secuencias de 16 elementos
 * @param inicio  Punto de partida
 * @param fin     Punto final
 */
void merge_sort(float *array, int inicio, int fin);

/**
 * Subfuncion de mergesort. Combina las subsoluciones del algoritmo
 * de ordenamiento por mezcla.
 *
 * @param array Es el array original en donde se copiaran las subsoluciones ordenadas.
 * @param inicio Es el inicio del primer subarray
 * @param medio Es el fin del primer subarray
 * @param fin Es el fin del segundo subarray
 */
void combinar(float *array, int inicio, int medio, int fin);
#endif
