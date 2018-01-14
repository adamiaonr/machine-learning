#!/bin/bash

# EM tests folder directory
EM_DIR="/home/adamiaonr/Documents/PhD/ml/tests/em/"

# create array with sizes for labeled data
sizes=( 20 40 100 300 500 800 1000 1260 )
#sizes=( 20 40 )

# number of runs
runs=20

ACC_RAW_EM=$EM_DIR"results.b.em.acc.raw"
ACC_RAW_NB=$EM_DIR"results.b.nb.acc.raw"

CONF_EM_TMP="conf.em.tmp"
CONF_EM_DEF_MIN=$EM_DIR"results.b.em.conf.min"
CONF_EM_DEF_MAX=$EM_DIR"results.b.em.conf.max"

CONF_NB_TMP="conf.nb.tmp"
CONF_NB_DEF_MIN=$EM_DIR"results.b.nb.conf.min"
CONF_NB_DEF_MAX=$EM_DIR"results.b.nb.conf.max"

# clear the results files
cat /dev/null > $EM_DIR"results.b.em.acc.raw"
cat /dev/null > $EM_DIR"results.b.nb.acc.raw"

cat /dev/null > $CONF_EM_DEF_MIN
cat /dev/null > $CONF_EM_DEF_MAX

cat /dev/null > $CONF_NB_DEF_MIN
cat /dev/null > $CONF_NB_DEF_MAX

# cycle through the elements in sizes
for size in ${sizes[@]}
do

    # add line to results files
    echo "SIZE= "$size >> $EM_DIR"results.b.em.acc.raw"
    echo "SIZE= "$size >> $EM_DIR"results.b.nb.acc.raw"
    echo "SIZE= "$size >> $CONF_EM_DEF_MIN
    echo "SIZE= "$size >> $CONF_EM_DEF_MAX
    echo "SIZE= "$size >> $CONF_NB_DEF_MIN
    echo "SIZE= "$size >> $CONF_NB_DEF_MAX

    # erase the contents of conf.*.tmp
    cat /dev/null > $CONF_EM_TMP
    cat /dev/null > $CONF_NB_TMP

    em_max="0"
    em_min="100.00"

    nb_max="0"
    nb_min="100.00"

    # run the EM tests for 10 times
    for (( i=1; i<=$runs; i++))
    do

        # number of elements per class to retrieve from the training set
        per_class=$(($size/20))

        # generate training set by randomly picking $per_class element per 
        # class from the training data only

        rainbow -d train-model/ --train-set=$per_class"pc" --print-doc-names=train > train.txt

        # remaining examples in training set should go to unlabeled dataset
        fgrep -vf train.txt full-train-set.txt > unlabeled.txt

        # run the classifier with EM
        rainbow -d model/ --method=em --em-print-perplexity=trainandunlabeled --train-set=train.txt --unlabeled-set=unlabeled.txt --em-stat-method=simple --test-set=full-test-set.txt --em-unlabeled-normalizer=1 --test=1 > results_em.txt

        # print results to a file
        cat results_em.txt | perl rainbow-stats | grep "percent accuracy" | grep -E -o '\<[0-9]{1,2}\.[0-9]{2,5}\>' >> $EM_DIR"results.b.em.acc.raw"

        # retrieve the last accuracy value
        accuracy=$(tail -1 $ACC_RAW_EM)

        # auxiliary variables
        aux_acc=$(echo $accuracy | sed 's/\.//')
        aux_em_min=$(echo $em_min | sed 's/\.//')
        aux_em_max=$(echo $em_max | sed 's/\.//')

        echo $aux_acc
        echo $aux_em_min
        echo $aux_em_max

        # check if it is the minimum value or maximum
        if [ $((aux_acc)) -lt $((aux_em_min)) ]; then

            em_min=$accuracy

            echo $em_min

        fi

        if [ $((aux_acc)) -gt $((aux_em_max)) ]; then

            em_max=$accuracy

            echo $em_max

        fi

        # save the confusion matrix results
        echo "ACC= "$accuracy >> $CONF_EM_TMP
        cat results_em.txt | perl rainbow-stats | grep "alt.atheism" -A 19 >> $CONF_EM_TMP

        # results for Naive Bayes only
        rainbow -d model/ --method=nb --train-set=train.txt --unlabeled-set=unlabeled.txt --test-set=full-test-set.txt --test=1 > results_nb.txt

        # print results to a file
        cat results_nb.txt | perl rainbow-stats | grep "percent accuracy" | grep -E -o '\<[0-9]{1,2}\.[0-9]{2,5}\>' >> $EM_DIR"results.b.nb.acc.raw"

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
    echo "END_SIZE= "$size >> $EM_DIR"results.b.em.acc.raw"
    echo "END_SIZE= "$size >> $EM_DIR"results.b.nb.acc.raw"

    #em_min='"'"ACC= "$em_min'"'
    #em_max='"'"ACC= "$em_max'"'

    #nb_min='"'"ACC= "$nb_min'"'
    #nb_max='"'"ACC= "$nb_max'"'

    echo $nb_max

    # take care of confusion matrices
    cat $CONF_EM_TMP | grep "ACC= $em_min" -A 20 >> $CONF_EM_DEF_MIN
    cat $CONF_EM_TMP | grep "ACC= $em_max" -A 20 >> $CONF_EM_DEF_MAX

    cat $CONF_NB_TMP | grep "ACC= $nb_min" -A 20 >> $CONF_NB_DEF_MIN
    cat $CONF_NB_TMP | grep "ACC= $nb_max" -A 20 >> $CONF_NB_DEF_MAX

    echo "END_SIZE= "$size >> $CONF_EM_DEF_MIN
    echo "END_SIZE= "$size >> $CONF_EM_DEF_MAX
    echo "END_SIZE= "$size >> $CONF_NB_DEF_MIN
    echo "END_SIZE= "$size >> $CONF_NB_DEF_MAX

done

exit
