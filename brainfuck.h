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
struct Expression_tag{
    ExpressionKind kind;
    union{
        int value;
        Expression_tag *block;
    } u;
    Expression_tag *next;
    Expression_tag *prev;
};
typedef struct Expression_tag Expression;

/*
 * Function prototype
 */
__global__ void kernel(char *res, char *data);
__device__ char brainfuck(char *source, int len);
__device__ Expression *parse(char **source, Token period);
__device__ char run(Expression *program);

#endif // BRAINFUCK_H_INCLUDED
