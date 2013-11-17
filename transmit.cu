#include "transmit.h"

__host__ Code *create_code(){
    Code *code = (Code *)malloc(sizeof(Code));
    code->next = NULL;
    return code;
}

__host__ Source *create_source(){
    Source *source = (Source *)malloc(sizeof(Source));
    source->codes = create_code();
    source->codes_len = 0;
    source->next = NULL;
    return source;
}

__host__ void deleteSource(Source *source){
    Source *next;
    Code *next_code;

    while(source){
        next = source->next;
        while(source->codes){
            next_code = source->codes->next;
            free(source->codes);
            source->codes = next_code;
        }
        free(source);
        source = next;
    }
}

__host__ Source *get_strings(FILE *in){
    Source *source = create_source();
    Source *cur_source = source;
    Source *prev_source = NULL;
    Code *code = cur_source->codes;
    int i = 0;
    char c;

    for(c = fgetc(in); c != EOF; c = fgetc(in)){
        if(c == '\n'){
            if(cur_source->codes_len > 0){
                code->code[i] = '\0';
                cur_source->next = create_source();
                prev_source = cur_source;
                cur_source = cur_source->next;
                code = cur_source->codes;
                i = 0;
            }
        } else{
            if(i == CODE_LENGTH){
                puts("overflow. malloc next.");
                code->next = create_code();
                code = code->next;
                i = 0;
            }
            code->code[i++] = c;
            cur_source->codes_len++;
        }
    }
    if(cur_source->codes_len == 0 && prev_source != NULL){
        prev_source->next = NULL;
        deleteSource(cur_source);
    }
    return source;
}

/*
 * [WARNING] String length must be shorter than 255.
 */
__host__ int pack_strings(char **data, Source *source){
    Source *cur_source;
    int source_len = 0, data_len = 0;
    char i;
    char *strhead, *lenhead;

    cur_source = source;
    while(cur_source){
        source_len++;
        data_len += cur_source->codes_len;
        cur_source = cur_source->next;
    }
    data_len += source_len * 2 + 1;

    *data = (char *)malloc(sizeof(char) * data_len);
    **data = source_len;
    lenhead = *data + 1;
    strhead = lenhead + source_len;

    while(source){
        *lenhead++ = source->codes_len + 1;
        while(source->codes){
            for(i = 0; i < CODE_LENGTH && source->codes->code[i]; i++){
                *strhead++ = source->codes->code[i];
            }
            *strhead++ = '\0';
            source->codes = source->codes->next;
        }
        source = source->next;
    }

    return data_len;
}

__host__ void transmit_data(char **data_d, char *data, int len){
    cudaMalloc(data_d, sizeof(char) * len);
    cudaMemcpy(*data_d, data, sizeof(char) * len, cudaMemcpyHostToDevice);
}
