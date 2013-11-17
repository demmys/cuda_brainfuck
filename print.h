#ifndef PRINT_H_INCLUDED
#define PRINT_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

extern const char *usage;

__host__ void error(char *format, ...);
__host__ void help();
__host__ void version();

#endif // PRINT_H_INCLUDED
