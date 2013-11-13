#ifndef BRAINFUCK_H_INCLUDED
#define BRAINFUCK_H_INCLUDED
#include "transmit.h"

/*
 * Enumerated type
 */
typedef enum{
    INC = 1,
    DEC,
    NEXT,
    PREV,
    PUT,
    GET,
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
};
typedef struct Expression_tag Expression;

/*
 * Function prototype
 */
__global__ void kernel(char *res, char *data);
__device__ char brainfuck(char *source, int len);
__device__ Token lex(char **source);

#endif // BRAINFUCK_H_INCLUDED
