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

from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from collections import defaultdict
from collections import OrderedDict
# for printing results
from prettytable import PrettyTable

from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    # read data
    data = np.loadtxt(open("students.csv", "rb"), delimiter=",", skiprows = 1)
    x = data[:, 1:]

    fig = plt.figure(figsize = (18, 5))

    # t-sne
    ax = fig.add_subplot(131)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    ax.set_title("""t-SNE""")

    print(x)

    tsne = manifold.TSNE(n_components = 2, init = 'pca', random_state = 0)
    mf = tsne.fit_transform(x)
    mf = pd.DataFrame(mf, columns = ['c1', 'c2'])

    # plot ids over 2D manifold
    ax.scatter(mf['c1'], mf['c2'], color = 'blue', marker = 'o', s = 10)

    k = 0
    for i, j in zip(mf['c1'].values, mf['c2'].values):
        corr = -0.05
        ax.annotate(str(k + 1),  xy = (i + corr, j + corr))
        k += 1

    # isomap
    ax = fig.add_subplot(132)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    ax.set_title("""isomap""")

    iso = manifold.Isomap(n_neighbors = 6, n_components = 2)
    iso.fit(x)
    # transform:
    #   - get k neighbors of new x
    #   - get distances to k neighbors + distances to every other point, 
    #     by adding the distance of each k neighbor to every other point
    #   - with this you get updated matrix M
    #   - from eigenvectors of modified version of M, get transformation
    mf = iso.transform(x)
    mf = pd.DataFrame(mf, columns = ['c1', 'c2'])

    # plot ids over 2D manifold
    ax.scatter(mf['c1'], mf['c2'], color = 'blue', marker = 'o', s = 10)

    k = 0
    for i, j in zip(mf['c1'].values, mf['c2'].values):
        corr = -0.05
        ax.annotate(str(k + 1),  xy = (i + corr, j + corr))
        k += 1

    # local linear embedding (lle)
    ax = fig.add_subplot(133)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    ax.set_title("""LLE""")

    lle = manifold.LocallyLinearEmbedding(n_neighbors = 12, n_components = 2, method = 'standard')
    lle.fit(x)
    # transform:
    #   - get k neighbors of new x
    #   - calculate weights between x and every other training point
    #   - update W
    #   - with modified version of W, compute eigenvalues and 
    #     d smallest eigenvectors, use them as transformation
    mf = lle.transform(x)
    mf = pd.DataFrame(mf, columns = ['c1', 'c2'])

    # plot ids over 2D manifold
    ax.scatter(mf['c1'], mf['c2'], color = 'blue', marker = 'o', s = 10)

    k = 0
    for i, j in zip(mf['c1'].values, mf['c2'].values):
        corr = -0.05
        ax.annotate(str(k + 1),  xy = (i + corr, j + corr))
        k += 1

    fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.3, hspace = None)
    plt.savefig("ex3.pdf", bbox_inches = 'tight', format = 'pdf')