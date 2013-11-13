#include "run.h"

__device__ char run(Expression *program){
    char ret = program->kind;
    while(program->next){
        program = program->next;
        free(program->prev);
    }
    free(program);
    return ret;
}
