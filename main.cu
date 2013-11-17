#include "main.h"

static int flags = 0;

__host__ int has_flags(char *mask_char, ...){
    va_list args;
    int p, q;

    va_start(args, mask_char);
    while(*mask_char){
        p = flags & va_arg(args, int);
        q = *mask_char++ - '0';
        if((p && !q) || (!p && q)){
            va_end(args);
            return 0;
        }
    }
    va_end(args);
    return 1;
}

__host__ void kernel_brainfuck(char **res, char *source, int source_len){
    char *source_d, *res_d;

    if(has_flags("11", F_TIME, F_MEMCPY_TIME)){
        stop_watch_start();
    }
    transmit_data(&source_d, source, source_len);

    cudaMalloc(&res_d, sizeof(char) * *source);
    if(has_flags("10", F_TIME, F_MEMCPY_TIME)){
        stop_watch_start();
    }
    kernel<<<1, *source>>>(res_d, source_d);
    if(has_flags("10", F_TIME, F_MEMCPY_TIME)){
        stop_watch_stop();
    }
    cudaFree(source_d);

    cudaMemcpy(*res, res_d, sizeof(char) * *source, cudaMemcpyDeviceToHost);
    cudaFree(res_d);
    if(has_flags("11", F_TIME, F_MEMCPY_TIME)){
        stop_watch_stop();
    }
}

__host__ void host_brainfuck(char **res, char *source){
    if(flags & F_TIME){
        stop_watch_start();
    }
    host(*res, source);
    if(flags & F_TIME){
        stop_watch_stop();
    }
}

__host__ int main(int argc, char *argv[]){
    extern int optind, optopt;
    extern int opterr;
    FILE *in;
    char c;
    Source *source;
    char *packed_source;
    int packed_source_len;
    int i;

    opterr = 0;
    while((c = getopt(argc, argv, "cdhmptv")) != -1){
        switch(c){
            case 'c':
                flags = flags | F_CPU;
                break;
            case 'd':
                flags = flags | F_DIGITAL;
                break;
            case 'h':
                help();
            case 'm':
                flags = flags | F_MEMCPY_TIME;
                break;
            case 'p':
                puts("option partition is not implemented yet.");
                break;
            case 't':
                flags = flags | F_TIME;
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
    /*
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
    */
    // <<<<< TEST
    packed_source_len = pack_strings(&packed_source, source);
    // TEST >>>>>
    /*
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
    */
    // <<<<< TEST
    char *res = (char *)malloc(sizeof(char) * *packed_source);
    if(flags & F_CPU){
        host_brainfuck(&res, packed_source);
    } else{
        kernel_brainfuck(&res, packed_source, packed_source_len);
    }
    for(i = 0; i < *packed_source; i++){
        if(flags & F_DIGITAL){
            printf("%d ", res[i]);
        } else{
            printf("%c", res[i]);
        }
    }
    if(flags & F_DIGITAL){
        puts("");
    }
    // TEST >>>>>
    /*
    for(i = 0; i < *packed_source; i++){
        if(res[i] < 33){
            printf("%d ", res[i]);
        } else{
            printf("%c", res[i]);
        }
    }
    */
    // <<<<< TEST
    if(flags & F_TIME){
        printf("\nReal run time: %10.6f (sec)\n", get_stop_watch_time());
    }

    return EXIT_SUCCESS;
}
