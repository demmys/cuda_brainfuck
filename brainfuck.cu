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
