#include "print.h"
#include "transmit.h"
#include "brainfuck.h"
#include <unistd.h>
#include <time.h>

#define THREAD_SIZE 14

static const int F_CPU = 0x01;
static const int F_TIME = 0x02;
static int flag = 0;

static time_t start, end;

__host__ void kernel_brainfuck(char *source, int source_len){
    // Host
    char res[THREAD_SIZE];
    // Device
    char *source_d, *res_d;

    transmit_data(&source_d, source, source_len);

    cudaMalloc(&res_d, sizeof(char) * THREAD_SIZE);
    if(flag & F_TIME){
        time(&start);
    }
    kernel<<<1, THREAD_SIZE>>>(res_d, source_d);
    if(flag & F_TIME){
        time(&end);
    }
    cudaFree(source_d);

    cudaMemcpy(res, res_d, sizeof(char) * THREAD_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(res_d);

    puts(res);
}

__host__ void host_brainfuck(char *source){
    char res[THREAD_SIZE];
    if(flag & F_TIME){
        time(&start);
    }
    host(res, source);
    if(flag & F_TIME){
        time(&end);
    }
    puts(res);
}

__host__ int main(int argc, char *argv[]){
    extern int optind, optopt;
    extern int opterr;
    FILE *in;
    char c;
    Source *source;
    char *packed_source;
    int packed_source_len;

    opterr = 0;
    while((c = getopt(argc, argv, "chtv")) != -1){
        switch(c){
            case 'c':
                flag = flag | F_CPU;
                break;
            case 'h':
                help();
            case 't':
                flag = flag | F_TIME;
                break;
            case 'v':
                version();
            default:
                error("illigal option \"%c\".\n%s\n", optopt, usage);
        }
    }
    argc -= optind;
    argv += optind;

    in = (argc > 0) ? fopen(argv[0], "r") : stdin;
    if(in == NULL){
        error("There is no file named \"%s\".\n", argv[0]);
    }

    source = get_strings(in);
    fclose(in);
    // TEST >>>>>
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
    // <<<<< TEST
    packed_source_len = pack_strings(&packed_source, source);
    // TEST >>>>>
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
    // <<<<< TEST
    if(flag & F_CPU){
        host_brainfuck(packed_source);
    } else{
        kernel_brainfuck(packed_source, packed_source_len);
    }

    if(flag & F_TIME){
        printf("time: %f, %f\n", start, end);
    }

    return EXIT_SUCCESS;
}
