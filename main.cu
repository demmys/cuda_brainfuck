#include "transmit.h"
#include "brainfuck.h"
#include <stdarg.h>

#define THREAD_SIZE 14

__host__ void kernel_brainfuck(char *source, int source_len){
    // Host
    char res[THREAD_SIZE];
    // Device
    char *source_d, *res_d;

    transmit_data(&source_d, source, source_len);

    cudaMalloc(&res_d, sizeof(char) * THREAD_SIZE);
    kernel<<<1, THREAD_SIZE>>>(res_d, source_d);
    cudaFree(source_d);

    cudaMemcpy(res, res_d, sizeof(char) * THREAD_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(res_d);

    puts(res);
}

__host__ void host_brainfuck(char *source){
    char res[THREAD_SIZE];
    host(res, source);
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
    Source *source;
    char *packed_source;
    int packed_source_len;

    in = (argc > 1) ? fopen(argv[argc - 1], "r") : stdin;
    if(in == NULL){
        error("\"%s\": no such file.\n", argv[argc - 1]);
    }

    source = get_strings(in);
    Source *source_tmp;
    Code *code;
    puts("\n");
    source_tmp = source;
    while(source){
        code = source->codes;
        while(code){
            puts(code->code);
            code = code->next;
        }
        printf("%d\n", source->codes_len);
        source = source->next;
    }
    source = source_tmp;
    packed_source_len = pack_strings(&packed_source, source);
    printf("\n\n%d\n\n", packed_source_len);
    int i;
    for(i = 0; i < packed_source_len; i++){
        if(packed_source[i] < 33){
            printf("%d ", packed_source[i]);
        } else{
            printf("%c", packed_source[i]);
        }
    }
    puts("");
    kernel_brainfuck(packed_source, packed_source_len);
    host_brainfuck(packed_source);

    fclose(in);

    return 0;
}
