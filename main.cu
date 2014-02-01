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

__host__ void kernel_brainfuck(int **res, int *source, int source_len, int block_size){
    int *source_d, *res_d;
    int grid_size = *source / block_size;
    if(grid_size * block_size < *source){
        grid_size++;
    }

    if(has_flags("11", F_TIME, F_MEMCPY_TIME)){
        stop_watch_start();
    }
    transmit_data(&source_d, source, source_len);

    cudaMalloc(&res_d, sizeof(int) * *source);
    if(has_flags("10", F_TIME, F_MEMCPY_TIME)){
        stop_watch_start();
    }
    kernel<<<*source / block_size, block_size>>>(res_d, source_d);
    cudaThreadSynchronize();
    if(has_flags("10", F_TIME, F_MEMCPY_TIME)){
        stop_watch_stop();
    }
    cudaFree(source_d);

    cudaMemcpy(*res, res_d, sizeof(int) * *source, cudaMemcpyDeviceToHost);
    cudaFree(res_d);
    if(has_flags("11", F_TIME, F_MEMCPY_TIME)){
        stop_watch_stop();
    }
}

__host__ void host_brainfuck(int **res, int *source){
    int thread_size = *source;
    int idx;
    ThreadArgs *args;
    pthread_t *threads;

    threads = (pthread_t *)malloc(sizeof(pthread_t) * thread_size);
    if(flags & F_TIME){
        stop_watch_start();
    }

    for(idx = 0; idx < thread_size; idx++){
        args = (ThreadArgs *)malloc(sizeof(ThreadArgs));
        args->res = *res;
        args->data = source;
        args->idx = idx;
        pthread_create(&threads[idx], NULL, host, (void *)args);
    }
    for(idx = 0; idx < thread_size; idx++){
        pthread_join(threads[idx], NULL);
    }

    if(flags & F_TIME){
        stop_watch_stop();
    }
    free(threads);
}

__host__ int main(int argc, char *argv[]){
    extern char *optarg;
    extern int optind, optopt, opterr;
    FILE *in;
    char c;
    Source *source;
    int *packed_source, *res;
    int packed_source_len;
    int i;
    char delimiter = '\n';
    int block_size = 1;

    opterr = 0;
    while((c = getopt(argc, argv, ":chlmntvb:d:")) != -1){
        switch(c){
            case 'b':
                block_size = atoi(optarg);
                break;
            case 'c':
                flags = flags | F_CPU;
                break;
            case 'd':
                delimiter = *optarg;
                break;
            case 'h':
                help();
            case 'l':
                flags = flags | F_LOG;
                break;
            case 'n':
                flags = flags | F_DIGITAL;
                break;
            case 'm':
                flags = flags | F_MEMCPY_TIME;
            case 't':
                flags = flags | F_TIME;
                break;
            case 'v':
                version();
            case ':':
                error("Option \"%c\" needs argument.\n%s\n", optopt, usage);
            default:
                error("Illigal option \"%c\".\n%s\n", optopt, usage);
        }
    }
    argc -= optind;
    argv += optind;

    in = (argc > 0) ? fopen(argv[0], "r") : stdin;
    if(in == NULL){
        error("There is no file named \"%s\".\n", argv[0]);
    }

    if((source = get_strings(in, delimiter)) == NULL){
        error("Nothing was inputted.\n");
    }
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
    printf("\n\npacked_source_len: %d\n\npacked_source: ", packed_source_len);
    for(i = 0; i < packed_source_len; i++){
        if(packed_source[i] < 33 || packed_source[i] > 127){
            printf("(%d)", packed_source[i]);
        } else{
            printf("%c", packed_source[i]);
        }
    }
    puts("");
    */
    // <<<<< TEST
    res = (int *)malloc(sizeof(int) * *packed_source);
    if(flags & F_CPU){
        host_brainfuck(&res, packed_source);
    } else{
        if(block_size < 1){
            block_size = 1;
        } else if(block_size > *packed_source){
            block_size = *packed_source;
        }
        kernel_brainfuck(&res, packed_source, packed_source_len, block_size);
    }

    if(flags & F_LOG){
        printf("%d,%-10.6f\n", *packed_source, get_stop_watch_time());
        return EXIT_SUCCESS;
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
    if(flags & F_TIME){
        printf("\nExecution time: %10.6f (sec)\n", get_stop_watch_time());
    }

    return EXIT_SUCCESS;
}
