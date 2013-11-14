#include "transmit.h"
#include "brainfuck.h"

#define THREAD_SIZE 14

__host__ int main(int argc, char *argv[]){
    // Host
    char *program[THREAD_SIZE] = {
        ">++++++++[<+++++++++>-]<.",
        ">++++++++++[<++++++++++>-]<+.",
        ">++++++++++[<+++++++++++>-]<--.",
        ">++++++++++[<+++++++++++>-]<--.",
        ">++++++++++[<+++++++++++>-]<+.",
        ">++++[<+++++++++++>-]<.",
        ">++++[<++++++++>-]<.",
        ">+++++++++[<++++++++++>-]<---.",
        ">++++++++++[<+++++++++++>-]<+.",
        ">++++++++++[<+++++++++++>-]<++++.",
        ">++++++++++[<+++++++++++>-]<--.",
        ">++++++++++[<++++++++++>-]<.",
        ">++++[<++++++++>-]<+.",
        "."
    };
    char *data = (char *)malloc(sizeof(char) * 512);
    int data_len;
    char *res;
    // Device
    char *data_d, *res_d;

    data_len = pack_strings(&data, program, THREAD_SIZE);

    transmit_data(&data_d, data, data_len);
    free(data);

    cudaMalloc(&res_d, sizeof(char) * THREAD_SIZE);
    kernel<<<1, THREAD_SIZE>>>(res_d, data_d);
    cudaFree(data_d);

    res = (char *)malloc(sizeof(char) * THREAD_SIZE);
    cudaMemcpy(res, res_d, sizeof(char) * THREAD_SIZE, cudaMemcpyDeviceToHost);
    puts(res);
    cudaFree(res_d);

    return 0;
}
