#ifndef LOOP_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <pthread.h>
#include "../stopwatch.h"

typedef enum{
    F_CPU = 0x01,
    F_LOG = 0x10
} Flag;

typedef struct{
    int idx;
    int *res;
} ThreadArgs;

#define LOOP_H_INCLUDED
#endif // LOOP_H_INCLUDED
