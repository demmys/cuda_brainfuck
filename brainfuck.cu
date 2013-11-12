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
            return Inc;
        case '-':
            return Dec;
        case '>':
            return Next;
        case '<':
            return Prev;
        case '.':
            return Put;
        case ',':
            return Get;
        case '[':
            return Begin;
        case ']':
            return End;
        case '\0':
            return EOP;
        default:
            return lex(source);
    }
}
