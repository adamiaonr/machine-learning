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

from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# for weight initialization
from random import seed
from random import random

def sigmoid(w, X):
    return 1 / (1 + np.exp(-np.dot(X, w)))

def calc_w(X, y, w, num_iters):

    # add column of 1s to X for w_0
    X = np.vstack((np.ones(len(X)), X.T)).T

    # iteratively update w num_iters times
    for i in xrange(num_iters):

        # S and H calculation
        S = sigmoid(w, X)
        S = np.diag((S * (1.0 - S)))
        H = np.dot(X.T, np.dot(S, X))
        # term within '[]' in slide 14
        a = np.dot(S, np.dot(X, w)) + y - sigmoid(w, X)
        # finally w
        w = np.dot(np.linalg.inv(H), np.dot(X.T, a))

    return w

def calc_error(X, y, w):
    res = np.where(sigmoid(w, X) > 0.5, 1.0, 0.0)
    return sum(np.absolute(y - res)) / len(X)


def get_ys(w, degree, x_range = [0, 30], w_1 = []):

    # plot lms regression results
    x = np.arange(x_range[0], x_range[1], 0.1)
    x = [x ** d for d in xrange(degree + 1)]
    x = np.vstack(x).T

    y = 0
    if len(w_1):
        x = np.vstack((x.T, (np.dot(x, (-w_1[:2]))) / w_1[2])).T
        y = (np.dot(x[:,:2], (-w[:2])) - (w[3] * sigmoid(w_1, x))) / w[2]
    else:
        y = (np.dot(x, (-w[:2]))) / w[2]

    return x[:,1], y

def derivative(a):
    return a * (1.0 - a)

if __name__ == "__main__":

    # read training data
    data = np.loadtxt(open("dataset.txt", "rb"), delimiter=",", skiprows = 0)

    # plot training data
    fig = plt.figure(figsize = (12, 5))
    ax = fig.add_subplot(121)

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    # plot original data
    colors = ['red', 'blue']
    markers = ['o', 'o']
    for i, case in enumerate([0, 1]):
        ax.scatter(data[np.where(data[:, 2] == case), 0], data[np.where(data[:, 2] == case), 1], facecolors = 'none', edgecolors = colors[i], s = 50, marker = markers[i], label = ('class %d' % (case)))

    # get x and y from 80% of the training data
    # we use np.random.choice() to get 80 random points
    # ix = np.random.choice(len(data), int(len(data) * 0.8))
    # ix = np.arange(0, 80, 1)
    X = data[:80, :2]
    X = np.vstack((np.ones(len(X)), X.T)).T
    y = data[:80, 2]

    # how will we represent our 'neural net'?
    #   - 1 input layer w/ 3 activation units, 1 for each inout x_0, x_1 and x_2
    #   - 1 hidden layer w/ 4 activation units:
    #       - 1 unit w/ sigmoid activation function, which acts as the 'first' block in the cascade
    #       - the 3 remaining units have a linear activation function which respectively pass x_0, x_1 and 
    #         x_2 'untouched' to the output layer
    #   - 1 output layer w/ 1 activation unit, which acts as the 'second' block in the cascade

    # for this particular example, i'll base my implementation in an iterative approach as:
    #   - https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    #   - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    
    # initialize neural network as list of dicts : each element is a layer 
    # (in our case, a block)
    nn = []
    # weigths for 1st block are w_0 (bias), w_1 and w_2 (for x_1 and x_2, respectively)
    nn.append({'w' : np.zeros(X.shape[1]), 'pd' : np.zeros(X.shape[1])})
    # weights for 2nd block are w_0, w_1, w_2 (as w/ the 1st block) AND w_3 for the output of block_1
    nn.append({'w' : np.zeros(X.shape[1] + 1), 'pd' : np.zeros(X.shape[1] + 1)})

    # parameters for gradient descent
    alpha = 0.0001
    n_iter = 100000

    for k in xrange(n_iter):
        for i, x in enumerate(X[:,:]):

            # 1) forward propagation phase
            a = defaultdict(np.array)
            # calculate activations: 
            #   - 1st block / (hidden layer)
            #       - 3 inputs (x_0, x_1, x_2) are passed 'untouched'
            #       - a sigmoid activation is passed to the output layer
            a[0] = np.concatenate([x, np.array([sigmoid(nn[0]['w'], x)])])
            #   - 2nd block (output layer)
            a[1] = sigmoid(nn[1]['w'], a[0].T)
                
            # 2) compute partial derivatives of weights, using backpropagation: 
            #   - for output layer
            delta = (a[1] - y[i])
            for j in xrange(nn[1]['w'].shape[0]):
                nn[1]['pd'][j] = delta * a[0][j]
            #   - for hidden layer
            delta = (delta * nn[1]['w'][3]) * derivative(a[0][3])
            for j in xrange(nn[0]['w'].shape[0]):
                nn[0]['pd'][j] = delta * x[j]

            # now, update weights using learning rate alpha
            for b in [0, 1]:
                for j in xrange(nn[b]['w'].shape[0]):
                     nn[b]['w'][j] = nn[b]['w'][j] - (alpha * (nn[b]['pd'][j]))

    # calculate training and test error of cascade model
    print("error cascade :\n\t[TRAINING] = %.5f" % (calc_error(np.vstack((X.T, sigmoid(nn[0]['w'], X))).T, y, nn[1]['w'])))
    _X = data[-20:, :2]
    _X = np.vstack((np.ones(len(_X)), _X.T)).T
    _X = np.vstack((_X.T, sigmoid(nn[0]['w'], _X))).T
    _y = data[-20:, 2]
    print("\t[TEST] = %.5f" % (1.0 - calc_error(_X, _y, nn[1]['w'])))

    # plot classifier output after cascade
    p = sigmoid(nn[1]['w'], _X)
    ax.scatter(_X[np.where(p >  0.5), 1], _X[np.where(p >  0.5), 2], color = 'blue', marker = '_', s = 20, label = ('casc. class %d' % (1)))
    ax.scatter(_X[np.where(p <= 0.5), 1], _X[np.where(p <= 0.5), 2], color = 'red',  marker = '_', s = 20, label = ('casc. class %d' % (0)))

    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_ylim(-20, 120)
    ax.legend(fontsize = 12, ncol = 2, loc = 'lower left')

    fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.3, hspace = None)
    plt.savefig("ex1-backpropagation.pdf", bbox_inches = 'tight', format = 'pdf')