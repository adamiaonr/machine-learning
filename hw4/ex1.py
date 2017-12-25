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

# FIXME: for some reason, this isn't working. needs revision.
class basic_svm:

    def __init__(self, d = 2, p = 2, C = 0.25):
        self.alphas = None
        self.svs = None
        self.b = 0.0

        self.p = p
        self.C = C

    def train(self, x, y):

        # solve the underlying quadratic programming problem
        # refer to http://goelhardik.github.io/2016/11/28/svm-cvxopt/ for details
        n = x.shape[0]
        d = x.shape[1]
        # convert np arrays to matrices
        X = np.matrix(x)
        Y = np.matrix(y)

        # P parameter of quadprog, an n x n matrix, w/ polynomial kernel (1 + x'*x)^d
        # FIXME: kernel function should be a parameter
        O = np.matrix(np.ones(n))
        O = np.dot(O.T, O)
        P = cvxopt.matrix(np.multiply(np.dot(Y.T, Y), (O + np.dot(X, X.T))**self.p))

        # q parameter 
        q = cvxopt.matrix(np.matrix(-np.ones((n, 1))))

        # bounds : G and h parameters
        # G and l have (u)pper and (l)ower bounds, more specifically 0 <= x <= C
        G_lb = np.matrix(-np.eye(n))
        h_lb = np.matrix(np.zeros((n, 1)))
        G_ub = np.matrix(np.eye(n))
        h_ub = self.C * np.matrix(np.ones((n, 1)))
        # concatenate lower and upper bounds in a single G and h
        G = cvxopt.matrix(np.vstack((G_lb, G_ub)))
        h = cvxopt.matrix(np.concatenate((h_lb, h_ub)))

        # Ax = b constraints
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(np.zeros(1))
        # run the cvxopt solver
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # extract alphas and support vectors
        # FIXME: for some reason, all alphas are above 1e-4. this can't
        # be right...
        alphas = np.array(sol['x'])
        self.alphas = alphas[(alphas > 1e-4).reshape(-1)]
        self.svs = x[(alphas > 1e-4).reshape(-1)]
        # calculate b
        w = np.zeros(len(self.svs))
        for i, x in enumerate(self.svs):
            w[i] = np.dot(self.alphas.T, (1.0 + np.dot(self.svs, x))**self.p)
        # not sure if np.mean() is the correct way
        self.b = np.mean(y[(alphas > 1e-4).reshape(-1)] - w)

    def predict(self, X, Y = None):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            y[i] = np.dot(self.alphas.T, (1.0 + np.dot(self.svs, x))**self.p) + self.b
        return y

def custom_mode(data):

    # training data split at 80%
    x = data[:80, :2]
    y = data[:80, 2]

    # since we want a one-vs-one multiclass svm, find pairs of 
    # basic svm classifiers: 
    #   - svm classifiers are saved in a dict, indexed 
    #     by the respective pair of classes. e.g. [C_1, C_2] 
    #     holds the classifier between classes C_1 and C_2. 
    #   - each svm classifier is a combination of w and b, 
    #     the parameters of the discriminant function.
    #
    svms = defaultdict()
    # get all pairwise combinations of classes, based 
    # on the unique classes in y
    pairs = list(itertools.combinations(set(y), 2))
    # train each of the pairwise svm classifiers
    for i, pair in enumerate(pairs):
        # fill class 0 of the pair (y = 1)
        _x = x[np.where(y == pair[0])]
        _y = np.ones(len(_x))
        # fill class 1 of the pair (y = -1)
        __x = x[np.where(y == pair[1])]
        _x = np.vstack((_x, __x))
        _y = np.concatenate((_y, -np.ones(len(__x))))

        # train pairwise svm classifier
        # svms[pair] = basic_svm()
        # svms[pair].train(_x, _y)


        # pred = svms[pair].predict(_x)
        # print(pred / np.absolute(pred))
        # print(_y)
        # print(np.dot(_x, svms[pair]['w'].T) + svms[pair]['b'])

if __name__ == "__main__":

    # read data
    data = np.loadtxt(open("myData.txt", "rb"), delimiter=";", skiprows = 0)

    # use an ArgumentParser for a nice CLI
    parser = argparse.ArgumentParser()
    # options (self-explanatory)
    parser.add_argument(
        "--mode", 
         help = """use '--mode custom' for custom svm implementation""")

    args = parser.parse_args()

    if args.mode == 'custom':
        custom_mode(data)
        sys.exit(0)

    # scikit learn svm implementation

    # split data into training, validation and test subsets
    x = defaultdict()
    y = defaultdict()
    _x, x['test'], _y, y['test'] = train_test_split(data[:, :2], data[:, 2], test_size = 0.33, train_size = 0.67)
    x['train'], x['val'], y['train'], y['val'] = train_test_split(_x, _y, test_size = 0.20, train_size = 0.80)

    # accuracy values saved in defaultdict
    accuracy = OrderedDict()
    for p in xrange(5):
        for c in [.25, .5, 1.0, 2.0, 4.0, 8.0]:

            # train
            model = SVC(kernel = 'poly', degree = p + 1, coef0 = 1.0, C = c)
            model.fit(x['train'], y['train'])

            # gather accuracies for dataset partitions
            accuracy[(p,c)] = {'train' : 0.0, 'val' : 0.0, 'test' : 0.0}
            for case in ['train', 'val', 'test']:
                y_pred = model.predict(x[case])
                aux = (y_pred == y[case])
                aux = np.sum(aux.astype(float), 0)
                accuracy[(p,c)][case] = aux / y[case].size

    table = PrettyTable(['p', 'C', 'train acc.', 'val acc.', 'test acc.'])
    for param in accuracy:
        table.add_row([
            ('%d' % (param[0] + 1)),
            ('%.2f' % (param[1])), 
            ('%.3f' % (accuracy[param]['train'])),
            ('%.3f' % (accuracy[param]['val'])),
            ('%.3f' % (accuracy[param]['test']))])

    print(table)
