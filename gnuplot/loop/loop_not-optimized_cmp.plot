call "shared_settings.plot" "$0"
set xrange [0:256]
set yrange [0:0.25]
set xtics 16
set mxtics 4
plot "cpu_loop_speed_not-optimized/average.log" linestyle 1 lc rgb "red" ti "8 core CPU", \
     "gpu_loop_speed_not-optimized/average.log" linestyle 1 lc rgb "blue" ti "768 core GPU"
