#ifndef BRAINFUCK_H_INCLUDED
#define BRAINFUCK_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <pthread.h>

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
typedef struct{
    char *res;
    char *data;
    int idx;
} ThreadArgs;

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

typedef struct{
    Expression *program;
    int header;
    int memory_size;
    int **memory;
} VirtualMachine;

/*
 * Function prototype
 */
__host__ void *host(void *args);
__global__ void kernel(char *res, char *data);
__host__ __device__ void thread(char *res, char *data, int idx);
__host__ __device__ char brainfuck(char *source, int len);
__host__ __device__ Expression *parse(char **source, Token period);
__host__ __device__ int run(Expression *program);

#endif // BRAINFUCK_H_INCLUDED
