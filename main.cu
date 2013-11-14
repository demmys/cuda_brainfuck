#include "transmit.h"
#include "brainfuck.h"

#define THREAD_SIZE 14

__host__ void kernel_brainfuck(char **program){
    // Host
    char *data = (char *)malloc(sizeof(char) * 512);
    int data_len;
    char res[THREAD_SIZE];
    // Device
    char *data_d, *res_d;

    data_len = pack_strings(&data, program, THREAD_SIZE);
    transmit_data(&data_d, data, data_len);
    free(data);

    cudaMalloc(&res_d, sizeof(char) * THREAD_SIZE);
    kernel<<<1, THREAD_SIZE>>>(res_d, data_d);
    cudaFree(data_d);

    cudaMemcpy(res, res_d, sizeof(char) * THREAD_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(res_d);

    puts(res);
}

__host__ void host_brainfuck(char **program){
    char *data = (char *)malloc(sizeof(char) * 512);
    char res[THREAD_SIZE];

    pack_strings(&data, program, THREAD_SIZE);
    host(res, data);
    free(data);

    puts(res);
}

__host__ int main(int argc, char *argv[]){
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
    kernel_brainfuck(program);
    host_brainfuck(program);

    return 0;
}
