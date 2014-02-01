call "shared_settings.plot" "$0"
set xrange [0:48]
set yrange [0:0.3]
set xtics 8
set mxtics 4
plot "cpu_speed_tick1/average.log" linestyle 1 ps 0.7 lc rgb "red" ti "8 core CPU"
