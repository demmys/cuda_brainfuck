#include "transmit.h"

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
