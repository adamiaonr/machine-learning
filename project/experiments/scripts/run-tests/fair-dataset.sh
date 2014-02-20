#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Usage: $0 [TRAINING EXAMPLES PER CLASS]"
    echo "E.g. $0 1"
    exit 1
fi

# generate training set by randomly picking $1 element per class from the 
# training data only
rainbow -d train-model/ --train-set=$1pc --print-doc-names=train > train.txt

# remaining examples in training set should go to unlabeled dataset
fgrep -vf train.txt full-train-set.txt > unlabeled.txt

# run the classifier with EM
rainbow -d model/ --method=em --em-compare-to-nb --em-print-perplexity=trainandunlabeled --train-set=train.txt --unlabeled-set=unlabeled.txt --em-stat-method=simple --test-set=full-test-set.txt --em-unlabeled-normalizer=100 --test=1 > results_em.txt

# print results
cat results_em.txt | perl rainbow-stats

rainbow -d model/ --method=nb --train-set=train.txt --unlabeled-set=unlabeled.txt --test-set=full-test-set.txt --test=1 > results_nb.txt

# print results
cat results_nb.txt | perl rainbow-stats

