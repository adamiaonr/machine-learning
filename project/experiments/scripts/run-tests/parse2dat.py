from datetime import date
from datetime import datetime
from decimal import Decimal
from time import mktime
import decimal
import time
import sys
import os
import collections

TIMESTAMP   = "TIMESTAMP"
MEM  		= "MEM"
MEM_TOTAL   = "MEM_FREE"
CPU       	= "CPU"

def main():

    if len(sys.argv) < 1:
        print "Incorrect usage: python parse2dat.py [.raw file]"
        return

    if not os.path.isfile(sys.argv[1]):
        print ".raw file '" + sys.argv[1] + "' does not exist. Exiting."
        return

    # open the .raw file
    RAW_FILE = open(sys.argv[1])
    raw_lines = RAW_FILE.readlines()
    RAW_FILE.close()

    dat_printList = []

    # initial line
    dat_printList.append("LABELED_SIZE" + "\t" + "ACCURACY")

    # cycle through raw_lines
    for line in raw_lines:

        # check if the start of a new size of labeled data
        if line.find("SIZE=") != -1:
            size_labeled = line.split()[1]

        elif line.find("END_SIZE=") != -1:

            # add a .dat line to printList
            mean_accuracy = sum_records / Decimal(num_records)
            dat_printList.append(size_labeled + "\t" + str(mean_accuracy))

            num_records = 0
            sum_records = 0

        else:

            # extract the value from the line
            record = Decimal(line.split()[0])

            num_records = num_records + 1
            sum_records = sum_records + record

            print record
            print sum_records

    #if os.path.isfile(sys.argv[3] + str("cpu-mem-stats.dat")):
        #print ".dat file already exists"
        #return

    gnuplot_dat_file = open((sys.argv[1]).replace(".raw",".dat"), "w+");

    for item in dat_printList:
        gnuplot_dat_file.write(item);

    gnuplot_dat_file.close();

if __name__ == "__main__":
    main()
