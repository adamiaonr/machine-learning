    # Gnuplot script file

set terminal postscript enhanced font "Helvetica, 16"
set output '| ps2pdf - em-lambda.pdf'

set tmargin 5
set bmargin 7
set lmargin 5
set rmargin 0

unset log                           # remove any log-scaling
unset label                         # remove any previous labels

set ylabel "Accuracy (%)"
set xlabel "Amount of Labeled Data"
set logscale x

set xtic auto nomirror                       # set xtics automatically
set ytic auto nomirror                       # set ytics automatically

set xrange[10:5000]

set key inside
set key right center

#set key outside horizontal
#set key center bottom

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

plot "../../tests/em-lambda/backup/results.c.em.acc.dat.1" using 1:2 title 'L = 1' with linespoints, \
    "../../tests/em-lambda/backup/results.c.em.acc.dat.2" using 1:2 title 'L = 2' with linespoints, \
    "../../tests/em-lambda/backup/results.c.em.acc.dat.5" using 1:2 title 'L = 5' with linespoints, \
    "../../tests/em-lambda/backup/results.c.em.acc.dat.10" using 1:2 title 'L = 10' with linespoints, \
    "../../tests/em-lambda/backup/results.c.em.acc.dat.50" using 1:2 title 'L = 50' with linespoints, \
    "../../tests/em-lambda/backup/results.c.em.acc.dat.100" using 1:2 title 'L = 100' with linespoints, \
    "../../tests/em-lambda/backup/results.c.nb.acc.dat" using 1:2 title 'MNB' with linespoints

