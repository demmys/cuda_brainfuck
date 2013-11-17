#include "stopwatch.h"

static StopWatch sw;

__host__ double get_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__host__ void stop_watch_start(){
    sw.start = get_time();
}

__host__ void stop_watch_stop(){
    sw.stop = get_time();
}

__host__ double get_stop_watch_time(){
    return sw.stop - sw.start;
}
