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
    
    # 1st stage

    # get x and y from 80% of the training data
    # we use np.random.choice() to get 80 random points
    # ix = np.random.choice(len(data), int(len(data) * 0.8))
    # ix = np.arange(0, 80, 1)
    X = data[:80, :2]
    y = data[:80, 2]

    # w has same dimensions as each data point + w_0
    w_1 = np.zeros(X.shape[1] + 1)
    # calculate w using Newton's method     
    w_1 = calc_w(X, y, w_1, 100)
    # accuracy of simple logistic regression (training / test)
    print("error simple :\n\t[TRAINING] = %.5f" % (calc_error(np.vstack((np.ones(len(X)), X.T)).T, y, w_1)))
    _X = data[-20:, :2]
    _y = data[-20:, 2]
    print("\t[TEST] = %.5f" % (1.0 - calc_error(np.vstack((np.ones(len(_X)), _X.T)).T, _y, w_1)))

    # plot classifier output after first block
    # get the output from 1st stage (sigmoid function), probabilities P(Y = 1 | x)
    p = sigmoid(w_1, np.vstack((np.ones(len(_X)), _X.T)).T)

    ax.scatter(_X[np.where(p >  0.5), 0], _X[np.where(p >  0.5), 1], color = 'blue', marker = '^', s = 10, label = ('simple class %d' % (1)))
    ax.scatter(_X[np.where(p <= 0.5), 0], _X[np.where(p <= 0.5), 1], color = 'red',  marker = '^', s = 10, label = ('simple class %d' % (0)))

    # print the original boundary region in green
    xx, yy = get_ys(w_1, 1, x_range = [20, 100])
    ax.plot(xx, yy, linewidth = 1.0, color = 'green', label = 'simple bound.')        

    # get the output from 1st stage (sigmoid function), probabilities P(Y = 1 | x)
    p = sigmoid(w_1, np.vstack((np.ones(len(X)), X.T)).T)

    # 2nd stage
    # add p as an extra feature to X (as X_2)
    X_2 = np.vstack((X.T, (p))).T
    # calculate w_2
    w_2 = np.zeros(X_2.shape[1] + 1)
    w_2 = calc_w(X_2, y, w_2, 100)
    print("error cascade :\n\t[TRAINING] = %.5f" % (calc_error(np.vstack((np.ones(len(X_2)), X_2.T)).T, y, w_2)))
    _X = data[-20:, :2]
    _X = np.vstack((_X.T, sigmoid(w_1, np.vstack((np.ones(len(_X)), _X.T)).T))).T
    _y = data[-20:, 2]
    print("\t[TEST] = %.5f" % (1.0 - calc_error(np.vstack((np.ones(len(_X)), _X.T)).T, _y, w_2)))

    print(w_1)
    print(w_2)

    # prediction for point [80, 41]
    test_point = np.array([80, 41])
    p = sigmoid(w_1, np.concatenate([np.array([1]), test_point]).T)
    print("classify(%s) :\n\t[SIMPLE] = %d (%.5f)" % (str(test_point), (1 if p > 0.5 else 0), p))
    test_point = np.concatenate([test_point, np.array([p])])
    p = sigmoid(w_2, np.concatenate([np.array([1]), test_point]).T)
    print("\t[CASCADE] = %d (%.5f)" % ((1 if p > 0.5 else 0), p))

    # plot classifier output after cascade
    p = sigmoid(w_2, np.vstack((np.ones(len(_X)), _X.T)).T)
    ax.scatter(_X[np.where(p >  0.5), 0], _X[np.where(p >  0.5), 1], color = 'blue', marker = '_', s = 20, label = ('casc. class %d' % (1)))
    ax.scatter(_X[np.where(p <= 0.5), 0], _X[np.where(p <= 0.5), 1], color = 'red',  marker = '_', s = 20, label = ('casc. class %d' % (0)))

    ax2 = fig.add_subplot(122)
    ax2.xaxis.grid(True)
    ax2.yaxis.grid(True)

    ax2.set_title("""output of 'simple' stage""")

    p = sigmoid(w_1, np.vstack((np.ones(len(data[-20:, :2])), data[-20:, :2].T)).T)
    ax2.scatter(_X[np.where(p >  0.5), 1], p[np.where(p >  0.5)], color = 'blue', marker = 'x', s = 20, label = ('casc. class %d' % (1)))
    ax2.scatter(_X[np.where(p <= 0.5), 1], p[np.where(p <= 0.5)], color = 'red',  marker = 'x', s = 20, label = ('casc. class %d' % (0)))

    ax2.set_xlabel("y")
    ax2.set_ylabel("logistic([x,y])")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks(np.arange(0.0, 1.1, .1))

    # # plot the cascade boundary region as a green dashed line
    # xx, yy = get_ys(w_2, 1, x_range = [20, 100], w_1 = w_1)
    # ax.plot(xx, yy, linewidth = 1.0, ls = '--', color = 'm', label = '2nd stage boundary')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-20, 120)
    ax.legend(fontsize = 12, ncol = 2, loc = 'lower left')

    fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.3, hspace = None)
    plt.savefig("ex1.pdf", bbox_inches = 'tight', format = 'pdf')