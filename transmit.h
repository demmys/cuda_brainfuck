#ifndef TRANSMIT_H_INCLUDED
#define TRANSMIT_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>

int pack_strings(char *data[], char *strs[], char len);
void transmit_data(char **data_d, char *data, int len);

#endif // TRANSMIT_H_INCLUDED
