#include "transmit.h"
#include "brainfuck.h"
#include "util.h"

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
