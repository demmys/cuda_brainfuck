#include "print.h"

const char *usage = "usage: gpubf [-chtv] [file ...]";

__host__ void error(char *format, ...){
    va_list args;

    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    exit(EXIT_FAILURE);
}

__host__ void help(void){
    puts(usage);
    exit(EXIT_SUCCESS);
}

__host__ void version(void){
    puts("Brainfuck interpreter on GPGPU version 1.0");
    exit(EXIT_SUCCESS);
}
