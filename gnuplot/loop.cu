#include "loop.h"

static int flags = 0;
const char *usage = "USAGE: gpuloop [-chl] [-t number of threads]";

__host__ __device__ void loop(int *res, int idx){
    int i;
    for(i = 0; i < 1000000000; i++);
    res[idx] = i;
}

__host__ void *host(void *args){
    loop(((ThreadArgs *)args)->res, ((ThreadArgs *)args)->idx);
    free((ThreadArgs *)args);

    return (void *)NULL;
}

__global__ void kernel(int *res, int thread_num){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < thread_num){
        loop(res, idx);
    }
}

__host__ int *host_loop(int thread_num){
    int *res;
    int idx;
    ThreadArgs *args;
    pthread_t *threads;

    res = (int *)malloc(sizeof(int) * thread_num);
    threads = (pthread_t *)malloc(sizeof(pthread_t) * thread_num);

    stop_watch_start();
    for(idx = 0; idx < thread_num; idx++){
        args = (ThreadArgs *)malloc(sizeof(ThreadArgs));
        args->idx = idx;
        args->res = res;
        pthread_create(&threads[idx], NULL, host, (void *)args);
    }
    for(idx = 0; idx < thread_num; idx++){
        pthread_join(threads[idx], NULL);
    }
    stop_watch_stop();

    free(threads);
    return res;
}

__host__ int *kernel_loop(int thread_num){
    int *res_d, *res;

    cudaMalloc(&res_d, sizeof(int) * thread_num);

    stop_watch_start();
    kernel<<<thread_num, 1>>>(res_d, thread_num);
    cudaThreadSynchronize();
    stop_watch_stop();

    res = (int *)malloc(sizeof(int) * thread_num);
    cudaMemcpy(res, res_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(res_d);
    return res;
}

__host__ int main(int argc, char *argv[]){
    extern char *optarg;
    extern int optind, optopt, opterr;
    char c;
    int thread_num = 1;
    int *res;
    int i;

    opterr = 0;
    while((c = getopt(argc, argv, ":chlt:")) != -1){
        switch(c){
            case 'c':
                flags = flags | 0x01;
                break;
            case 'h':
                puts(usage);
                exit(EXIT_SUCCESS);
            case 'l':
                flags = flags | 0x01;
                break;
            case 't':
                thread_num = atoi(optarg);
                break;
            case ':':
                fprintf(stderr, "Option \"%c\" needs argument.\n%s\n", optopt, usage);
                exit(EXIT_FAILURE);
            default:
                fprintf(stderr, "Illigal option \"%c\".\n%s\n", optopt, usage);
                exit(EXIT_FAILURE);
        }
    }

    if(flags & F_CPU){
        res = host_loop(thread_num);
    } else{
        res = kernel_loop(thread_num);
    }

    if(flags & F_LOG){
        printf("%d,%-10.6f\n", thread_num, get_stop_watch_time());
    } else{
        for(i = 0; i < thread_num; i++){
            printf("%d ", res[i]);
        }
        printf("\nExecution time: %10.6f (sec)\n", get_stop_watch_time());
    }
    free(res);

    return EXIT_SUCCESS;
}
