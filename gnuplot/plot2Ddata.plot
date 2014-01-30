set datafile separator ","
set xtics 64
set mxtics 4
set style line 1 lt 1 lw 2 lc rgb "blue"
set style line 2 lt 1 lw 2 lc rgb "red"
plot "cpu_speed/average.log" with lines linestyle 2 ti "8 core CPU", \
     "gpu_speed/average.log" with lines linestyle 1 ti "768 core GPU"
set terminal postscript eps enhanced color
set output "average.eps"
replot
set output
set terminal x11
