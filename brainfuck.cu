#include "brainfuck.h"

__global__ void kernel(char *res, char *data){
    int idx = threadIdx.x;
    int phead = *data + 1;
    int i;

    for(i = 0; i < idx; i++){
        phead += data[i + 1];
    }
    res[idx] = brainfuck(data + phead, data[idx]);
}

__host__ int main(int argc, char *argv[]){
    // Host
    char *program[2] = { ">++++[<++++++++>-]<+.", ">++++[<++++++++>-]<+." };
    char *data = (char *)malloc(sizeof(char) * 64);
    int data_len;
    char *res;
    // Device
    char *data_d, *res_d;

    data_len = pack_strings(&data, program, 2);
    show_data(data);

    transmit_data(&data_d, data, data_len);
    free(data);

    cudaMalloc(&res_d, sizeof(char) * 2);
    kernel<<<1, 2>>>(res_d, data_d);
    cudaFree(data_d);

    res = (char *)malloc(sizeof(char) * 3);
    cudaMemcpy(res, res_d, sizeof(char) * 2, cudaMemcpyDeviceToHost);
    res[2] = '\0';
    show_data(res);
    cudaFree(res_d);

    return 0;
}

__device__ char brainfuck(char *program, int len){
    return *program;
}

__host__ void show_data(char *data){
    for(; *data; data++){
        if(*data > 32){
            putchar(*data);
        } else{
            printf("%d ", *data);
        }
    }
    puts("");
}
