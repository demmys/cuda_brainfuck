set datafile separator ","
plot "cpu_speed/average.log"
plot "gpu_speed/average.log"
set terminal postscript eps enhanced color
set output "average.eps"
replot
set output
set terminal x11
