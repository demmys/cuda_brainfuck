set terminal postscript eps enhanced color
set output "$1"
set datafile separator ","
set xrange [0:48]
set yrange [0:0.3]
set xtics 8
set mxtics 4
set style line 1 pt 7 ps 1
plot "$0" linestyle 1 lc rgb "red" ti "8 core CPU"
