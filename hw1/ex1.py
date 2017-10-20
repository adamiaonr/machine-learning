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

# calculates euclidean norm of rows in a
def get_euclidean_norm(a):
    return np.sqrt(np.sum(a ** 2, axis = 1))

def get_training_error(prediction_class_1, prediction_class_2):
    prediction_class_1 = np.array(prediction_class_1)
    prediction_class_2 = np.array(prediction_class_2)
    return (1.0 / (len(prediction_class_1) + len(prediction_class_2))) * 0.5 * (sum(np.absolute(prediction_class_1 - (-1))) + sum(np.absolute(prediction_class_2 - 1)))

def get_interclass_var(data_class_1, data_class_2):
    dim = data_class_1.shape[data_class_1.ndim - 1]
    mean_c2 = np.mean(data_class_2, axis = 0) * np.ones((len(data_class_1), dim))
    return (1.0 / len(data_class_1)) * np.sum((data_class_1 - mean_c2) ** 2, axis = 0)

def meanPrediction(data_class_1, data_class_2, data_unknown_class):

    # save predictions on hash table : [<data-point-str>] -> [<class #>]
    predictions = defaultdict()
    # calculate dimensions of data
    dim = data_class_1.shape[data_class_1.ndim - 1]
    # plot data scatter plot (sanity check)
    if dim == 3:

        fig = plt.figure()
        # add subplots for 3d and 2d plots (all axis combinations)
        ax_xyz, ax_xy, ax_xz, ax_yz = fig.add_subplot(221, projection = '3d'), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)

        colors = ['red', 'blue']
        markers = ['o', '^']
        for i, data in enumerate([data_class_1, data_class_2]):
            # 3d plot
            ax_xyz.scatter(data[:,0], data[:,1], data[:,2], color = colors[i], marker = markers[i])
            # 2d plots
            ax_xy.scatter(data[:,0], data[:,1], color = colors[i], marker = markers[i])
            ax_xz.scatter(data[:,0], data[:,2], color = colors[i], marker = markers[i])
            ax_yz.scatter(data[:,1], data[:,2], color = colors[i], marker = markers[i])

        # axis labels
        # 3d plot
        ax_xyz.set_xlabel('XX')
        ax_xyz.set_ylabel('YY')
        ax_xyz.set_zlabel('ZZ')
        # 2d plots
        ax_xy.set_xlabel('XX')
        ax_xy.set_ylabel('YY')

        ax_xz.set_xlabel('XX')
        ax_xz.set_ylabel('ZZ')

        ax_yz.set_xlabel('YY')
        ax_yz.set_ylabel('ZZ')

        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
        plt.savefig("ex1.pdf", bbox_inches = 'tight', format = 'pdf')

    # we try 2 classification methods : 
    #   1) dist. of test data to each of the classes' mean (i.e. mean of training data)
    #   2) mean of dist. of test data to each of the training samples
    #
    # ... and choose the shortest distance

    # method 1 : faster, O(n), where n is size of test set
    
    # ** training phase ** : determine mean of each class
    means = [np.mean(data_class_1, axis = 0), np.mean(data_class_2, axis = 0)]
    variances = [np.var(data_class_1, axis = 0), np.var(data_class_2, axis = 0)]
    # ** test phase **
    # calculate euclidean distances to means of classes
    # dists_to_mean = [get_euclidean_norm(data_unknown_class - (means[0] * np.ones((len(data_unknown_class), dim)))), \
    #     get_euclidean_norm(data_unknown_class - (means[1] * np.ones((len(data_unknown_class), dim))))]
    dists_to_mean = [get_euclidean_norm(data_unknown_class - means[0]), get_euclidean_norm(data_unknown_class - means[1])]
    # predictions should be a 1 x n array, w/ elements in {-1, 1}: -1 for class 1, 1 for class 2
    predictions_1 = dists_to_mean[0] - dists_to_mean[1]
    # normalize to get domain in {-1 (class 1), 1 (class 2)}
    predictions_1 = (predictions_1 / np.absolute(predictions_1))

    # method 2 : slower (classif. time complexity O(n * t), where t is size of training set)
    # for each test point, calculate euclidean distance to each training point, use mean of
    # distances to classify
    predictions_2 = []
    for data_u in data_unknown_class:
        # data_u_v = [data_u * np.ones((len(data_class_1), dim)), data_u * np.ones((len(data_class_2), dim))]
        # distances = [get_euclidean_norm(data_u_v[0] - data_class_1), get_euclidean_norm(data_u_v[1] - data_class_2)]
        distances = [get_euclidean_norm(data_class_1 - data_u), get_euclidean_norm(data_class_2 - data_u)]
        # prediction for sample data_u calculated from the mean of distances
        # print("mean euclidean dist : %f, %f" % (np.mean(distances[0], axis = 0), np.mean(distances[1], axis = 0)))
        predictions = np.mean(distances[0], axis = 0) - np.mean(distances[1], axis = 0)
        predictions_2.append(predictions / abs(predictions))

    return [predictions_1, predictions_2]

if __name__ == "__main__":

    # prediction type : 
    #   0 - dist. to mean, 
    #   1 - mean of euclidean dist. to training samples (as asked in ex1)
    prediction_type = 1

    if prediction_type == 0:
        print("using euclidean dist. to mean")
    else:
        print("using mean of euclidean dist.")

    # read training data : classes 1 and 2 from .csv files
    data_class_1 = np.loadtxt(open("data-class-1.csv", "rb"), delimiter=",", skiprows = 0)
    data_class_2 = np.loadtxt(open("data-class-2.csv", "rb"), delimiter=",", skiprows = 0)

    # statistics of training data
    print("data class 1 stats:")
    print("\tMEAN: %s" % (str(np.mean(data_class_1, axis = 0))))
    print("\tVARIANCE: %s" % (str(np.var(data_class_1, axis = 0))))
    print("\tINTER-CLASS VARIANCE: %s" % (str(get_interclass_var(data_class_1, data_class_2))))

    print("\ndata class 2 stats:")
    print("\tMEAN: %s" % (str(np.mean(data_class_2, axis = 0))))
    print("\tVARIANCE: %s" % (str(np.var(data_class_2, axis = 0))))
    print("\tINTER-CLASS VARIANCE: %s\n" % (str(get_interclass_var(data_class_2, data_class_1))))

    # generate random unknown data to be classified
    data_unknown_class = np.random.rand(20, 3) * 20.0 - 10.0

    # ex1.0 : run meanPrediction on data_unknown_class
    print("ex1.0 : test data (%dD)" % (data_unknown_class.shape[data_unknown_class.ndim - 1]))
    p = meanPrediction(data_class_1, data_class_2, data_unknown_class)
    print("output : %s" % (p[prediction_type]))

    # ex1.1 : run meanPrediction on data_class_1 and data_class_2 (only x1)
    #         to determine training error
    print("ex1.1 : training error (%dD)" % (data_class_1[:,:1].shape[data_class_1[:,:1].ndim - 1]))

    p1 = meanPrediction(data_class_1[:,:1], data_class_2[:,:1], data_class_1[:,:1])
    p2 = meanPrediction(data_class_1[:,:1], data_class_2[:,:1], data_class_2[:,:1])
    print("training error: %f" % (get_training_error(p1[prediction_type], p2[prediction_type])))

    # ex1.2 : run meanPrediction on data_class_1 and data_class_2 (only x1,x2)
    #         to determine training error
    print("ex1.2 : training error (%dD)" % (data_class_1[:,:2].shape[data_class_1[:,:2].ndim - 1]))
    p1 = meanPrediction(data_class_1[:,:2], data_class_2[:,:2], data_class_1[:,:2])
    p2 = meanPrediction(data_class_1[:,:2], data_class_2[:,:2], data_class_2[:,:2])
    print("training error: %f" % (get_training_error(p1[prediction_type], p2[prediction_type])))

    # ex1.2 : run meanPrediction on data_class_1 and data_class_2 (only x1,x2)
    #         to determine training error
    print("ex1.3 : training error (%dD)" % (data_class_1.shape[data_class_1.ndim - 1]))
    p1 = meanPrediction(data_class_1, data_class_2, data_class_1)
    p2 = meanPrediction(data_class_1, data_class_2, data_class_2)
    print("training error: %f" % (get_training_error(p1[prediction_type], p2[prediction_type])))