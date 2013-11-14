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
    return run(parse(&source, EOP));
}

/*
 * compile
 */
__device__ void appendExpression(Expression *head, Expression *append){
    append->prev = head->prev;
    append->prev->next = append;
    head->prev = append;
}

/*
 * Atom expression
 * - INC
 * - DEC
 * - NEXT
 * - PREV
 * - GET
 * - PUT
 */
__device__ Expression *createAtomExpression(ExpressionKind kind, int value){
    Expression *ex = (Expression *)malloc(sizeof(Expression));
    ex->kind = kind;
    ex->u.value = value;
    ex->next = NULL;
    ex->prev = NULL;
    return ex;
}
__device__ void addAtomExpression(Expression **head, ExpressionKind kind, int value){
    if(*head == NULL){
        *head = createAtomExpression(kind, value);
        (*head)->prev = *head;
        return;
    }
    if((*head)->prev->kind == kind){
        (*head)->prev->u.value += value;
    } else{
        appendExpression(*head, createAtomExpression(kind, value));
    }
}

/*
 * While expression
 * - BEGIN
 */
__device__ Expression *createWhileExpression(){
    Expression *ex = (Expression *)malloc(sizeof(Expression));
    ex->kind = WHILE_EXPRESSION;
    ex->u.block = NULL;
    ex->next = NULL;
    ex->prev = NULL;
    return ex;
}
__device__ void addWhileExpression(Expression **head){
    if(*head == NULL){
        *head = createWhileExpression();
        (*head)->prev = *head;
        return;
    }
    appendExpression(*head, createWhileExpression());
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

__device__ Expression *parse(char **source, Token period){
    Token token;
    Expression *head = NULL;

    while((token = lex(source)) != period){
        switch(token){
            case INC:
                addAtomExpression(&head, ADD_EXPRESSION, 1);
                break;
            case DEC:
                addAtomExpression(&head, ADD_EXPRESSION, -1);
                break;
            case NEXT:
                addAtomExpression(&head, MOVE_EXPRESSION, 1);
                break;
            case PREV:
                addAtomExpression(&head, MOVE_EXPRESSION, -1);
                break;
            case GET:
                addAtomExpression(&head, GET_EXPRESSION, 1);
                break;
            case PUT:
                addAtomExpression(&head, PUT_EXPRESSION, 1);
                break;
            case BEGIN:
                addWhileExpression(&head);
                head->prev->u.block = parse(source, END);
        }
    }
    return head;
}

/*
 * run
 */
__device__ void deleteProgram(Expression *program){
    while(program->next){
        program = program->next;
        free(program->prev);
    }
    free(program);
}

__device__ char run(Expression *program){
    char ret = program->kind;
    deleteProgram(program);
    return ret;
}
