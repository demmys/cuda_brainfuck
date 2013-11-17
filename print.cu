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
    puts("\noptions:");
    puts("\t-c\tRun same program on 1 core 1 thread CPU.");
    puts("\t-d\tDisplay result of execution in digit.");
    puts("\t-h\tDisplay available options and exit.");
    puts("\t-m\tWhen use with -t option, display real run time includes memory copy time between host and device after the execution.");
    puts("\t-t\tDisplay real run time after the execution.");
    puts("\t-v\tDisplay product version and exit.");
    exit(EXIT_SUCCESS);
}

__host__ void version(void){
    puts("Brainfuck interpreter on GPGPU version 1.0");
    exit(EXIT_SUCCESS);
}
