#include <stdlib.h>
#include <malloc.h>
#include "main.h"
#include "sort_simd.h"

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		print_err("Error: Falta el nombre del archivo\n");
		exit(EXIT_FAILURE);
	}

	float *data __attribute__((aligned(16)));
	float *stdata;
	__m128 *regs;

	// se abre y "pesa" el archivo
	FILE* input_file;
	input_file = fopen(argv[1], "r");

	if (input_file == NULL)
	{
		print_err("Error: no existe el archivo de entrada «%s»", argv[1]);
	}

	size_t sof = fsof(input_file);
	size_t n_elem = sof/4;
	size_t n_reg = n_elem/4;

	// se reserva memoria segun tamagno del archivo

	data = (float*) malloc(n_elem * sizeof(float));
	regs = (__m128*) malloc(n_reg * sizeof(__m128));

	if (data== NULL || regs == NULL)
	{
		print_err("Error: no se pudo reservar memoria");
		exit(EXIT_FAILURE);
	}

	size_t leidos = fread(data, sizeof(float), n_elem, input_file);

	fclose(input_file);

	printf("leidos: %lu\n", leidos); 

	cargar_registros(data, regs, n_elem);

	ordenar_registros(regs, n_reg);

	guardar_registros(data, regs, n_reg);

	free(regs);
	stdata = (float*) malloc(n_elem * sizeof(float));

	printf("Haciendo el MWMS\n");
	
	mw_merge_sort(stdata, data, n_elem);

	char fname[100];
	FILE *salida;

	sprintf(fname, "%s.sorted.raw", argv[1]);
	salida = fopen(fname, "w"); 

	fwrite(stdata, sizeof(float), n_elem, salida);

	fclose(salida);
	free(data);
	free(stdata);

	/*
	for (int i=0; i< n_elem; i++)
	{
		for (int j=0; j< 16; j++)
			printf("%f ", stdata[i]);
		printf("\n");
	}
	*/

	return 0; 
}

void print_err(const char * format, ...)
{
	char temp[100];
	va_list args;
	va_start(args, format); 

	sprintf(temp, "\e[0;31m%s\e[0m", format);

	vfprintf(stderr, temp, args);

	va_end(args);
}

void ordenar_registros(__m128 * lreg, size_t size)
{
	__m128 *offset = lreg;

	for (int i=0; i< size; i+=4)
	{
		sort_SIMD(offset, offset+1, offset+2, offset+3);
		offset += 4;
	}
}

void cargar_registros(float *src, __m128 *dst, size_t ndatos)
{
	float *offset = src;
	size_t nreg = ndatos/4;

	for (int i=0; i< nreg; i++)
	{
		dst[i] = _mm_load_ps(offset);
		offset += 4;
	}
}

void guardar_registros(float *dst, __m128 *src, size_t nreg)
{
	float *offset = dst;

	for (int i=0; i< nreg; i++)
	{
		_mm_store_ps(offset, src[i]);
		offset += 4;
	}
}

void print_arr(float list[], size_t lenght)
{
	printf("[");
	for (int i=0; i< lenght; i++)
		printf("%2.1f ", list[i]);
	printf("]");
}

void print_m128(__m128 reg)
{
	float r[4] __attribute__((aligned(16)));

	_mm_store_ps(r, reg);

	print_arr(r, 4);
}

void printf_m128(char fmsg[], __m128 reg, ...)
{
	va_list pargs;
	va_start(pargs, reg);

	vprintf(fmsg, pargs);
	print_m128(reg);
	printf("\n");

	va_end(pargs); 
}

long fsof(FILE * ptrFILE)
{
	long size;
	long current;

	if (ptrFILE == NULL) return -1;

	current = ftell(ptrFILE);

	fseek(ptrFILE, 0, SEEK_END);
	size = ftell(ptrFILE);
	fseek(ptrFILE, current, SEEK_SET);

	return size;
}
