#ifndef STOPWATCH_H_INCLUDED
#define STOPWATCH_H_INCLUDED
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

typedef struct{
    double start;
    double stop;
} StopWatch;

__host__ void stop_watch_start();
__host__ void stop_watch_stop();
__host__ double get_stop_watch_time();

#endif // STOPWATCH_H_INCLUDED
