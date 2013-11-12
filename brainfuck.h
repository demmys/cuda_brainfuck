#ifndef BRAINFUCK_H_INCLUDED
#define BRAINFUCK_H_INCLUDED
#include "transmit.h"

/*
 * Enumerated type
 */
typedef enum{
    Inc = 1,
    Dec,
    Next,
    Prev,
    Put,
    Get,
    Begin,
    End,
    EOP
} Token;

typedef enum{
    Atom,
    Block
} StatementKind;

/*
 * Structure
 */
struct Statement_tag{
    StatementKind kind;
    Statement_tag *child;
};
typedef struct Statement_tag Statement;

/*
 * Function prototype
 */
__global__ void kernel(char *res, char *data);
__device__ char brainfuck(char *source, int len);
__device__ Token lex(char **source);

#endif // BRAINFUCK_H_INCLUDED
