
#ifndef BRAINFUCK_H_INCLUDED
#define BRAINFUCK_H_INCLUDED
#include "transmit.h"

__device__ char brainfuck(char *program, int len);

__host__ void show_data(char *data);

#endif // BRAINFUCK_H_INCLUDED
