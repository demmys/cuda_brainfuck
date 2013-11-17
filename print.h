#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

extern const char *usage;

__host__ void error(char *format, ...);
__host__ void help();
__host__ void version();
