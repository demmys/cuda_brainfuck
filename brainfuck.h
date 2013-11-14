#ifndef BRAINFUCK_H_INCLUDED
#define BRAINFUCK_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>

/*
 * Enumerated type
 */
typedef enum{
    INC,
    DEC,
    NEXT,
    PREV,
    GET,
    PUT,
    BEGIN,
    END,
    EOP
} Token;

typedef enum{
    ADD_EXPRESSION,
    MOVE_EXPRESSION,
    GET_EXPRESSION,
    PUT_EXPRESSION,
    WHILE_EXPRESSION
} ExpressionKind;

/*
 * Structure
 */
typedef struct Expression_tag Expression;
struct Expression_tag{
    ExpressionKind kind;
    union{
        int value;
        Expression *block;
    } u;
    Expression *next;
    Expression *prev;
};

/*
typedef struct Memory_tag Memory;
struct Memory_tag{
    int cell;
    Memory *next;
    Memory *prev;
};
typedef struct{
    Memory *memory;
} VirtualMachine;
*/
typedef struct{
    Expression *program;
    int header;
    int memory_size;
    int **memory;
} VirtualMachine;

/*
 * Function prototype
 */
__global__ void kernel(char *res, char *data);
__device__ char brainfuck(char *source, int len);
__device__ Expression *parse(char **source, Token period);
__device__ int run(Expression *program);

#endif // BRAINFUCK_H_INCLUDED
