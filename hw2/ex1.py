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

def gradient_descent_costs(X, y, costs, w, alpha, num_iters):

    # add column of 1s to X for w_0
    X = np.vstack((np.ones(len(X)), X.T)).T

    # iteratively update w num_iters times
    for i in xrange(num_iters):
        w = w + alpha * np.dot(costs * (y - np.dot(w,X.T)), X)

    return w

def get_ys(w):

    # plot lms regression results
    x = np.arange(0, 30, 0.1)
    x = [x ** d for d in xrange(degree + 1)]
    x = np.vstack(x).T
    y = np.dot(x, w.T)

    return x[:,1], y

if __name__ == "__main__":

    # read training data
    rocket_heights = np.loadtxt(open("rocket-heights.txt", "rb"), delimiter=",", skiprows = 0)
    print("x,y = \n%s" % (rocket_heights[:,:2]))

    # prepare X for degree 2 poly regression
    degree = 2
    X = [rocket_heights[:,0] ** d for d in np.arange(1, degree + 1)]
    X = np.vstack(X).T

    ## find w w/ original costs
    # initialize w to 0s
    w = np.zeros(X.shape[1] + 1)
    costs = rocket_heights[:,2]
    w = gradient_descent_costs(X, rocket_heights[:,1], costs, w, 0.000001, 100000)
    print("w for 1 cost : %s" % (w))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot original data
    ax.scatter(rocket_heights[:,0], rocket_heights[:,1], color = 'blue', marker = 'o')
    # plot model output
    x, y = get_ys(w)
    ax.scatter(x, y, color = 'red', marker = '.', s = 5)

    ## find w w/ other costs which 'force' point (10.0, 1073.0)
    # initialize w to 0s
    w = np.zeros(X.shape[1] + 1)
    costs = [0.1, 10.0, 0.1, 0.1, 0.1, 0.1]
    w = gradient_descent_costs(X, rocket_heights[:,1], costs, w, 0.000001, 100000)
    print("w for other cost : %s" % (w))
    # plot model output
    x, y = get_ys(w)
    ax.scatter(x, y, color = 'green', marker = '.', s = 5)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    plt.savefig("ex1.pdf", bbox_inches = 'tight', format = 'pdf')