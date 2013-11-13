#include "brainfuck.h"

__global__ void kernel(char *res, char *data){
    int idx = threadIdx.x;
    int phead = *data + 1;
    int i;

    for(i = 0; i < idx; i++){
        phead += data[i + 1];
    }
    res[idx] = brainfuck(data + phead, data[idx + 1]);
}

__device__ char brainfuck(char *source, int len){
    return lex(&source);
}

__device__ Token lex(char **source){
    switch(*(*source)++){
        case '+':
            return INC;
        case '-':
            return DEC;
        case '>':
            return NEXT;
        case '<':
            return PREV;
        case '.':
            return PUT;
        case ',':
            return GET;
        case '[':
            return BEGIN;
        case ']':
            return END;
        case '\0':
            return EOP;
        default:
            return lex(source);
    }
}

__device__ Expression *parse(char *source){
    Token token;
    Expression *ex, *parsing;

    while((token = lex(source)) != EOP){
        switch(token){
            case INC:
                break;
            case DEC:
                break;
            case NEXT:
                break;
            case PREV:
                break;
            case PUT:
                break;
            case GET:
                break;
            case BEGIN:
                break;
            case END:
        }
    }
}

__device__ Expression *createComputeExpression(ExpressionKind kind, int value){
    Expression *ex = malloc(sizeof(Expression));
    ex->kind = kind;
    ex->u.value = value;
    ex->next = NULL;
}

__device__ Expression *createIOExpression(ExpressionKind kind){
    Expression *ex = malloc(sizeof(Expression));
    ex->kind = kind;
    ex->u = NULL;
    ex->next = NULL;
}

__device__ Expression *createWhileExpression(){
    Expression *ex = malloc(sizeof(Expression));
    ex->kind = WHILE_EXPRESSION;
    ex->u.block = NULL;
    ex->next = NULL;
}
