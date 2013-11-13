#ifndef COMPILE_H_INCLUDED
#define COMPILE_H_INCLUDED
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
    ADD_EXPRESSION = 10,
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
__device__ Expression *parse(char **source, Token period);

#endif // COMPILE_H_INCLUDED
