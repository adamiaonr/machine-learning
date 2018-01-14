import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import os
import argparse
import sys
import glob
import math
import gmplot
import time
# for quadratic problem solving
import cvxopt
# for 'one vs. one' svm pairwise combinations
import itertools

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from collections import defaultdict
from collections import OrderedDict
# for printing results
from prettytable import PrettyTable

from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    # hmm model:
    #   - hidden states: climb (1), didn't climb (0)
    #   - observations : injured (0), not injured (1)
    #   - transition probabilities (between hidden states)
    #       1 0
    #     1 x y
    #     0 w z
    T = np.array([  [0.40, 0.60], 
                    [0.90, 0.10]])
    #   - start probabilities
    pi0 = np.array([0.98, 0.02])[:,None]
    #   - emission probabilities
    Px = np.array([ [0.8, 0.2], 
                    [0.1, 0.9]])

    # solving the problem:                              S M T W
    #   - we know that the sequence of observations is: *,0,0,1
    #   - note that we don't know if he got injured on sunday, so 
    #     there are 2 alternative observation sequences: 0,0,0,1 and 1,0,0,1
    
    #   - we want to know the prob of climbing on wed, i.e. 
    #     P(y_(wed)^(climb) = 1 | *,0,0,1)
    #   - according to slide 40/60, P(y_t^k = 1 | X) = P(y_t^k = 1, X) / P(x) 
    #                                                = (alpha_t^k * beta_t^k) / P(X)
    #
    #     alpha_t^k = P(x_1, x_2, ..., x_t, y_t^k = 1), i.e. forward probability
    #     beta_t^k  = P(x_(t+1), ..., x_T | y_t^k = 1), i.e. backward probability.
    #
    #   - note that in this case we don't need backward probability, since 
    #     the forward probability already considers the full observation
    #     sequence

    print("hw5::ex2 :")

    # observation sequences
    joint = 0.0
    marginal = 0.0
    for seq in ['1001', '0001']:
        obs_seq = [int(i) for i in seq]
        obs_seq = np.array(obs_seq)[:,None]
        # print (obs_seq)

        # forward method - slide 31
        alpha = Px[:, obs_seq[0]] * pi0
        # print(pi0.shape)
        # print(Px[:, obs_seq[0]])
        # print(Px[:, obs_seq[0]].shape)
        # print(alpha)
        # print(alpha.shape)

        for t in range(1, np.size(obs_seq)):
            alpha = Px[:, obs_seq[t]] * np.dot(T.T, alpha)

        # prob of the sequence of observations
        seq_prob = np.sum(alpha, 0)
        print("SEQUENCE %s:" % (seq))
        print("\tP([%s]) : %.3f" % (seq, seq_prob))
        # probability of climbing on wed
        print("\tP([%s], C) : %.3f" % (seq, alpha[0]))

        # update joint and marginal probabilities
        marginal += seq_prob
        joint += alpha[0]

    print("\nFINAL:")
    print("P([%s]) : %.3f" % ('*001', marginal))
    # probability of climbing on wed
    print("P(C|[%s]) : %.3f" % ('*001', joint / marginal))

    # alternative method:
    #   - calculate new start probabilities for monday
    #   - evaluate sequence 001 only
    pi0 = np.dot(T.T, pi0)

    obs_seq = [int(i) for i in '001']
    obs_seq = np.array(obs_seq)[:,None]

    # forward method - slide 31
    alpha = Px[:, obs_seq[0]] * pi0
    for t in range(1, np.size(obs_seq)):
        alpha = Px[:, obs_seq[t]] * np.dot(T.T, alpha)

    # prob of the sequence of observations
    seq_prob = np.sum(alpha, 0)

    print("\nhw5::ex2 :")
    print("\tP([%s]) : %.3f" % ('001', seq_prob))
    # probability of climbing on wed
    print("\tP(C|[%s]) : %.3f" % ('001', alpha[0] / seq_prob))
