#ifndef MAIN_H_INCLUDED
#define MAIN_H_INCLUDED
#include "print.h"
#include "stopwatch.h"
#include "transmit.h"
#include "brainfuck.h"
#include <unistd.h>

typedef enum{
    F_CPU = 0x01,
    F_TIME = 0x02,
    F_MEMCPY_TIME = 0x04,
    F_DIGITAL = 0x08
} Flag;

#endif // MAIN_H_INCLUDED
