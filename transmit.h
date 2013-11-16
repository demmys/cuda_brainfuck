#ifndef TRANSMIT_H_INCLUDED
#define TRANSMIT_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>

#define CODE_LENGTH 256

typedef struct Code_tag Code;
struct Code_tag{
    char code[CODE_LENGTH];
    Code *next;
};

typedef struct Source_tag Source;
struct Source_tag{
    Code *codes;
    int codes_len;
    Source *next;
};

__host__ Source *get_strings(FILE *in);
int pack_strings(char *data[], char *strs[], char len);
void transmit_data(char **data_d, char *data, int len);

#endif // TRANSMIT_H_INCLUDED
