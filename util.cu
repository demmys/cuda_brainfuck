#include "util.h"

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
