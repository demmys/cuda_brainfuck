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

__device__ void appendExpression(Expression *head, Expression *append){
    append->prev = head->prev;
    append->prev->next = append;
    head->prev = append;
}

/*
 * compile
 */
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

/*
__device__ Memory *createMemory(){
    Memory *mem = malloc(sizeof(Memory));
    mem->cell = 0;
    mem->next = NULL;
    mem->prev = NULL;
    return mem;
}

__device__ void expandMemory(VirtualMachine *vm){
    Memory *mem = createMemory();
    mem->prev = vm->memory->prev;
    vm->memory->prev->next = mem;
    vm->memory->prev = mem;
}

__device__ VirtualMachine *createVirtualMachine(){
    VirtualMachine *vm = malloc(sizeof(VirtualMachine));
    vm->memory = createMemory();
    vm->memory->prev = vm->memory;
}
*/
#define MEM_GRID_X 10
#define DEFAULT_MEM_SIZE 20
__device__ VirtualMachine *createVM(Expression *program){
    int i, j;
    VirtualMachine *vm = malloc(sizeof(VirtualMachine));

    vm->program = program;
    vm->header = 0;
    reallocVMMemory(vm, DEFAULT_MEM_SIZE);

    return vm;
}

__device__ void deleteExpression(Expression *ex){
    while(ex->next){
        ex = ex->next;
        free(ex->prev);
    }
    free(ex);
}

__device__ void deleteVM(VirtualMachine *vm){
    int i;
    int y = vm->memory_size / MEM_GRID_X + 1;
    for(i = 0; i < y; i++){
        free(vm->memory[i])
    }
    free(vm->memory);
    deleteExpression(vm->program);
    free(vm);
}

__device__ void reallocVMMemory(VirtualMachine *vm, int size){
    int required_y = (size - 1) / MEM_GRID_X + 1;
    int *new_memory[] = malloc(sizeof(int *) * required_y);

    for(i = 0; i < required_y; i++){
        if(vm->memory && vm->memory_size >= (i + 1) * MEM_GRID_X){
            new_memory[i] = vm->memory[i];
        } else{
            new_memory[i] = malloc(sizeof(int) * MEM_GRID_X);
            for(j = 0; j < MEM_GRID_X; j++){
                new_memory[i][j] = 0;
            }
        }
    }
    vm->memory = new_memory;
    vm->memory_size = size;
}

__device__ int *seekCurrentVMMemory(VirtualMachine *vm){
    int y = vm->header / MEM_GRID_X;
    int x = vm->header - MEM_GRID_X * y;
    return vm->memory[y] + x;
}

__device__ void addVMMemory(VirtualMachine *vm, int increment){
    *seekCurrentVMMemory(vm) += increment;
}

__device__ void moveVMHeader(VirtualMachine *vm, int increment){
    vm->header += increment;
    if(vm->header >= vm->memory_size){
        reallocVMMemory(vm, header + 1);
    }
}

__device__ int getVMValue(VirtualMachine *vm){
    return *seekCurrentVMMemory(vm);
}

__device__ int run(Expression *program){
    return runVM(createVM(program), 0);
}

__device__ int runVM(VirtualMachine *vm, int ret){
    Expression *jumped;

    while(vm->program != NULL){
        switch(vm->program->kind){
            case ADD_EXPRESSION:
                addVMMemory(vm, vm->program->u.value);
                break;
            case MOVE_EXPRESSION:
                moveVMHeader(vm, vm->program->u.value);
                break;
            case GET_EXPRESSION:
                ret = getVMValue(vm);
                break;
            case PUT_EXPRESSION:
                break;
            case WHILE_EXPRESSION:
                if(getVMValue(vm) != 0){
                    jumped = vm->program;
                    vm->program = vm->program->u.block;
                    ret = runVM(vm, ret);
                    vm->program = jumped;
                    continue;
                }
        }
        vm->program = vm->program->next;
    }

    return ret;
}
