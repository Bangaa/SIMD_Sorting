#ifndef __MAIN_H__
#define __MAIN_H__

#include <stdio.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <stdarg.h>


void print_arr(float list[], size_t lenght);
void print_m128(__m128 reg);
void printf_m128(char msg[], __m128 reg, ...) __attribute__((format(printf, 1, 3)));

/**
 * Dice el tamaño del archivo del descriptor ptrFILE.
 * @param ptrFILE Es el puntero que apunta al archivo ya abierto
 * @return El tamaño del archivo en bytes si es que no hubo problemas, o -1 si
 * el puntero no ha sido inicializado
 */
long fsof(FILE * ptrFILE);

void cargar_registros(float *src, __m128 *dst, size_t ndatos);
void ordenar_registros(__m128 * lreg, size_t size);
void guardar_registros(float *dst, __m128 *src, size_t nreg);
void print_err(const char * format, ...) __attribute__((format(printf, 1, 2)));
#endif
