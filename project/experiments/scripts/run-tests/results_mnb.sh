#!/bin/bash

# EM tests folder directory
MNB_DIR="/home/adamiaonr/Documents/PhD/ml/tests/mnb/"

# create array with sizes for labeled data
sizes=( 20 50 100 500 1000 5000 7520 )

# number of runs
runs=2

ACC_RAW_NB=$MNB_DIR"results.a.nb.acc.raw"

CONF_NB_TMP="conf.nb.tmp"
CONF_NB_DEF_MIN=$MNB_DIR"results.a.nb.conf.min"
CONF_NB_DEF_MAX=$MNB_DIR"results.a.nb.conf.max"

# clear the results files
cat /dev/null > $ACC_RAW_NB

cat /dev/null > $CONF_NB_DEF_MIN
cat /dev/null > $CONF_NB_DEF_MAX

# cycle through the elements in sizes
for size in ${sizes[@]}
do

    # add line to results files
    echo "SIZE= "$size >> $ACC_RAW_NB
    echo "SIZE= "$size >> $CONF_NB_DEF_MIN
    echo "SIZE= "$size >> $CONF_NB_DEF_MAX

    # erase the contents of conf.*.tmp
    cat /dev/null > $CONF_NB_TMP

    nb_max="0"
    nb_min="100.00"

    # run the MNB tests for 10 times
    for (( i=1; i<=$runs; i++))
    do

        # number of elements per class to retrieve from the training set
        per_class=$(($size/20))

        # generate training set by randomly picking $per_class element per 
        # class from the training data only

        rainbow -d train-model/ --train-set=$per_class"pc" --print-doc-names=train > train.txt

        # remaining examples in training set should go to unlabeled dataset
        #fgrep -vf train.txt full-train-set.txt > unlabeled.txt

        # results for Naive Bayes only
        rainbow -d model/ --method=nb --train-set=train.txt --test-set=full-test-set.txt --test=1 > results_nb.txt

        # print results to a file
        cat results_nb.txt | perl rainbow-stats | grep "percent accuracy" | grep -E -o '\<[0-9]{1,2}\.[0-9]{2,5}\>' >> $ACC_RAW_NB

        # retrieve the last accuracy value
        accuracy=$(tail -1 $ACC_RAW_NB)

        # auxiliary variables
        aux_acc=$(echo $accuracy | sed 's/\.//')
        aux_nb_min=$(echo $nb_min | sed 's/\.//')
        aux_nb_max=$(echo $nb_max | sed 's/\.//')

        # check if it is the minimum value or maximum
        if [ $((aux_acc)) -lt $((aux_nb_min)) ]; then

            nb_min=$accuracy

        fi

        if [ $((aux_acc)) -gt $((aux_nb_max)) ]; then

            nb_max=$accuracy

        fi

        # save the confusion matrix results
        echo "ACC= "$accuracy >> $CONF_NB_TMP
        cat results_nb.txt | perl rainbow-stats | grep "alt.atheism" -A 20 >> $CONF_NB_TMP

    done

    # add line to results files
    echo "END_SIZE= "$size >> $ACC_RAW_NB

    #em_min='"'"ACC= "$em_min'"'
    #em_max='"'"ACC= "$em_max'"'

    #nb_min='"'"ACC= "$nb_min'"'
    #nb_max='"'"ACC= "$nb_max'"'

    echo $nb_max

    # take care of confusion matrices

    cat $CONF_NB_TMP | grep "ACC= $nb_min" -A 20 >> $CONF_NB_DEF_MIN
    cat $CONF_NB_TMP | grep "ACC= $nb_max" -A 20 >> $CONF_NB_DEF_MAX

    echo "END_SIZE= "$size >> $CONF_NB_DEF_MIN
    echo "END_SIZE= "$size >> $CONF_NB_DEF_MAX

done

exit
