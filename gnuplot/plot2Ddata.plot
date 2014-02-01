set terminal postscript eps enhanced color
set output "$2"
set datafile separator ","
set xrange [0:256]
set yrange [0:16]
set xtics 16
set mxtics 4
set style line 1 pt 7 ps 0.3
plot "$0" linestyle 1 lc rgb "red" ti "8 core CPU", \
     "$1" linestyle 1 lc rgb "blue" ti "768 core GPU"
