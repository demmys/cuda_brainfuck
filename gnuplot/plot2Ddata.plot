set datafile separator ","
plot "$1.dat"
set terminal postscript eps enhanced color
set output "$1.eps"
replot
set output
set terminal x11
