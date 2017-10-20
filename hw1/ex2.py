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

def regression(x, y, degree):

    # adjust X[i] to [x_i^(0), x_i^(1), ..., x_i^(d - 1), x_i^(d)], as it's probably missing x_i^(0)
    x = [x ** d for d in xrange(degree + 1)]
    x = np.vstack((x[0], x[1])).T

    # calculate w using normal equation method in matrix form
    w = np.linalg.inv(np.dot(x.T, x))
    w = np.dot(w, x.T)
    w = np.dot(w, y)

    return w

def predict(x, w, min_charge_time, max_batt_runtime):

    # adjust x by adding a column of '1's
    x = np.vstack((np.ones(len(x)), x)).T
    # apply linear part
    y = np.dot(x,w.T)
    # for the indeces for which x > min_charge_time, set y to max_batt_runtime
    y[np.where(x[:,1] > min_charge_time)] = max_batt_runtime

    return y

if __name__ == "__main__":

    # read training data
    training_data = np.loadtxt(open("TabletTrainingdata.txt", "rb"), delimiter=",", skiprows = 0)

    # batt runtimes are described by a linear part + constant part
    # we use regression to find the slope of the linear part and 
    # as such we should only feed the training data relative to the linear 
    # part to regression()

    # to find the limits of the linear part, let's analyze the 
    # training data, i.e. find:
    #   - find the max. battery runtime : this the bat's max. capacity
    #   - find the min. charging time for which runtime is max. capacity
    max_batt_runtime = np.max(training_data[:,1])
    min_charge_time = np.min(training_data[training_data[:,1] == max_batt_runtime])

    # after finding min_charge_time, feed training data w/ x < min_charge_time to 
    # regression()
    training_data_linear = training_data[training_data[:,0] < min_charge_time]
    w = regression(training_data_linear[:,0], training_data_linear[:,1], 1)

    print(w)
    print(predict([1.0, 2.0, 8.0, 7.0, 54.0], w, min_charge_time, max_batt_runtime))