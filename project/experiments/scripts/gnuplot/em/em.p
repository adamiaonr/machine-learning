# Gnuplot script file

set terminal postscript enhanced font "Helvetica, 16"
set output '| ps2pdf - em.pdf'

set tmargin 5
set bmargin 7
set lmargin 5
set rmargin 0

unset log                           # remove any log-scaling
unset label                         # remove any previous labels

set ylabel "Accuracy (%)"
set xlabel "Amount of Labeled Data"

set xtic auto nomirror                       # set xtics automatically
set logscale x
set ytic auto nomirror                       # set ytics automatically
set yrange [0:80]

#set key outside horizontal
#set key center bottom

set key inside
set key right center

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

plot "../../tests/em/results.b.em.acc.dat" using 1:2 title 'MNB + EM' with linespoints, \
    "../../tests/em/results.b.nb.acc.dat" using 1:2 title 'MNB' with linespoints

