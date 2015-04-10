#include <getopt.h>
#include <stdlib.h>
#include <malloc.h>
#include "main.h"
#include "sort_simd.h"

const char usage[] =
			"opciones ...\n"
			"\n"
			"opciones:\n"
			"  -i <entrada>\n"
			"     Archivo binario con la lista de entrada desordenados\n"
			"  -o <salida>\n"
			"     Archivo binario de salida con la lista ordenada\n"
			"  -N <largo>\n"
			"     Largo de la lista. No es obligatorio ya que el programa lo deduce segun el tamagno del\n"
			"     archivo de entrada\n"
			"  -d <debug_level>\n"
			"     Cuando es 0 no se imprime nada por stdout. Cuando el valor es 1 se imprime la secuencia\n"
			"     final por stdout, un elemento por línea\n";

int main(int argc, char* argv[])
{
	int opt;
	char finpname[100] = "";
	char foupname[100] = "";
	int debug = 0;
	unsigned int numelem = 0;	//<! largo de la lista a ordenar

	if (argc == 1)
	{
		printf("uso: %s %s", argv[0], usage);
		exit(EXIT_FAILURE);
	}

	while ((opt = getopt(argc, argv, ":i:o:N:d:")) != -1)
	{
		switch (opt)
		{
			case 'i':
				sscanf(optarg, "%s", finpname);
				break;
			case 'o':
				sscanf(optarg, "%s", foupname);
				break;
			case 'N':
				sscanf(optarg, "%u", &numelem);
				break;
			case 'd':
				sscanf(optarg, "%i", &debug);
				break; 
		}
	} 

	float *data __attribute__((aligned(16)));
	float *stdata;
	__m128 *regs;

	// se abre y "pesa" el archivo
	FILE *file_in, *file_out;
	file_in = fopen(finpname, "r");

	if (file_in == NULL)
	{
		print_err("Error: no existe el archivo de entrada «%s»", finpname);
		exit(EXIT_FAILURE);
	}

	size_t sof = fsof(file_in);
	if (numelem == 0) 
		numelem = sof / 4;

	size_t numreg = numelem/4;

	// se reserva memoria segun tamagno del archivo

	data = (float*) malloc(numelem * sizeof(float));
	regs = (__m128*) malloc(numreg * sizeof(__m128));

	if (data== NULL || regs == NULL)
	{
		print_err("Error: no se pudo reservar memoria");
		exit(EXIT_FAILURE);
	}

	size_t leidos __attribute__((unused)) = fread(data, sizeof(float), numelem, file_in);
	fclose(file_in); 

	cargar_registros(data, regs, numelem);
	ordenar_registros(regs, numreg);
	guardar_registros(data, regs, numreg);

	free(regs);
	stdata = (float*) malloc(numelem * sizeof(float));

	printf("Haciendo el MWMS\n");
	
	mw_merge_sort(stdata, data, numelem);

	// SE ESCRIBE EL ARREGLO ORDENADO EN EL ARCHIVO DE SALIDA

	file_out = fopen(foupname, "w"); 

	fwrite(stdata, sizeof(float), numelem, file_out);

	if (debug == 1)
		for (int i=0; i< numelem; i++)
		{
			printf("%f\n", stdata[i]);
		}

	fclose(file_out);
	free(data);
	free(stdata); 

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
