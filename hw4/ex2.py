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
from sklearn.model_selection import train_test_split

from collections import defaultdict
from collections import OrderedDict
# for printing results
from prettytable import PrettyTable

from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    # read data
    data = np.loadtxt(open("wine.txt", "rb"), delimiter=",", skiprows = 0)
    x = data[:, 1:]
    y = data[:, 0]

    # calculate co-variance matrix of x
    S = np.cov(x.T)
    print(S.shape)

    # calculate eigenvalues and eigenvectors of S
    w, v = np.linalg.eig(S)
    # sort eigen-x by eigenvalues
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]

    # pick k largest eigenvalues (and respective eigenvectors),
    # i.e. first k principal components, get projection matrix W,
    # which transforms a 13-D space into a k-D space (hence size 13 x k, 
    # so that it transforms 1 x 13 arrays into 1 x k arrays), 
    # which is just the concatenation of k eigenvectors.
    k = 3
    W = v[:, :k]

    # project data on new k-D subspace (plot it)
    _x = np.dot(x, W)

    fig = plt.figure(figsize = (12, 10))
    ax_xyz, ax_xy, ax_xz, ax_yz = fig.add_subplot(221, projection = '3d'), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)

    ax_xy.xaxis.grid(True)
    ax_xy.yaxis.grid(True)
    ax_xz.xaxis.grid(True)
    ax_xz.yaxis.grid(True)
    ax_yz.xaxis.grid(True)
    ax_yz.yaxis.grid(True)

    colors = ['red', 'green', 'blue']
    for i, c in enumerate(set(y)):
        # 3-d
        ax_xyz.scatter(_x[np.where(y == c), 0], _x[np.where(y == c), 1], _x[np.where(y == c), 2], color = colors[i], marker = 'o', s = 20, label = ('%d' % (c)))
        # 2-d
        ax_xy.scatter(_x[np.where(y == c), 0], _x[np.where(y == c), 1], color = colors[i], marker = 'o', s = 20, label = ('%d' % (c)))
        ax_xz.scatter(_x[np.where(y == c), 0], _x[np.where(y == c), 2], color = colors[i], marker = 'o', s = 20, label = ('%d' % (c)))
        ax_yz.scatter(_x[np.where(y == c), 1], _x[np.where(y == c), 2], color = colors[i], marker = 'o', s = 20, label = ('%d' % (c)))

    # axis labels
    # 3d plot
    ax_xyz.set_xlabel('x')
    ax_xyz.set_ylabel('y')
    ax_xyz.set_zlabel('z')
    # 2d plots
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    ax_xz.set_xlabel('x')
    ax_xz.set_ylabel('z')
    ax_yz.set_xlabel('y')
    ax_yz.set_ylabel('z')

    fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.3, hspace = None)
    plt.savefig("ex2.pdf", bbox_inches = 'tight', format = 'pdf')

    # explained variance
    table = PrettyTable(['eigen value #', 'explained variance'])
    total = sum(w)
    for i, ev in enumerate(w):
        table.add_row([
            ('%d' % (i + 1)),
            ('%.2f' % ((ev / total) * 100.0))])

    print(table)
