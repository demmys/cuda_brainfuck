#include "transmit.h"
#include "brainfuck.h"
#include <stdarg.h>

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

__host__ void error(char *format, ...){
    va_list args;

    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    exit(EXIT_FAILURE);
}

__host__ int main(int argc, char *argv[]){
    FILE *in;
    int i = 0;
    int len = 20;
    char **source = (char **)malloc(sizeof(char *) * len);

    in = argc > 1 ? fopen(argv[1], "r") : stdin;
    if(!in){
        error("\"%s\": no such file.\n", argv[1]);
    }
    do{
        source[i] = (char *)malloc(sizeof(char) * 256);
        fgets(source[i++], 255, in);
        len++;
    } while(line != EOF);
    fclose(in);

    /*
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
    */
    kernel_brainfuck(program);
    host_brainfuck(program);

    return 0;
}
