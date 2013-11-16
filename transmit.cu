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

__host__ Source *get_strings(FILE *in){
    Source *source = create_source();
    Source *cur_source = source;
    Code *code = cur_source->codes;
    char c;
    int i;

    for(c = fgetc(in), i = 0; c != EOF; c = fgetc(in)){
        if(c == '\n'){
            cur_source->next = create_source();
            cur_source = cur_source->next;
            code = cur_source->code;
        } else{
            if(i == CODE_LENGTH){
                code->next = create_code();
                code = code->next;
                i = 0;
            }
            code->code[i++] = c;
            cur_source->codes_len++;
        }
    }
    return source;
}

__host__ char *pack_strings(int *data_len, FILE *fp){
    int c;
    char line_count = 0;
    char line_len = 0;
    char *data = malloc(sizeof(char) * 512);

    *data_len = 0;
    while((c = fgetc(in)) != EOF){
        if(c == '\n'){
            line_count++;
            line_len = 0;
        } else{
            *data_len++;
            line_len++;
        }
    }
}

/*
 * [WARNING] String length must be shorter than 255.
 *
 * @param::dist   data : *(char<fold(+, map(length, strs)) + len>)
 * @param::source strs : { char[], char[], ... }
 * @param::source len  : length(strs)
 */
__host__ int pack_strings(char *data[], char *strs[], char len){
    char i, j;
    char *strhead, *lenhead;
    int data_len;

    **data = len;
    lenhead = *data + 1;
    strhead = lenhead + len;
    data_len = len + 1;

    for(i = 0; i < len; i++){
        for(j = 0; strs[i][j]; j++){
            *strhead++ = strs[i][j];
        }
        *strhead++ = '\0';
        *lenhead++ = j + 1;
        data_len += j + 1;
    }

    return data_len;
}

__host__ void transmit_data(char **data_d, char *data, int len){
    cudaMalloc(data_d, sizeof(char) * len);
    cudaMemcpy(*data_d, data, sizeof(char) * len, cudaMemcpyHostToDevice);
}
