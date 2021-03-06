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
from sklearn.neighbors.kde import KernelDensity

def train(training_data):

    # priors of genders
    prior = defaultdict(float)

    # histograms for heights
    height_hist = defaultdict()
    height_hist[1] = defaultdict(float)  # nr. males
    height_hist[2] = defaultdict(float)  # nr. females
    # kde for weight
    prob_weight_given_gender = defaultdict()
    
    # calculate priors of genders
    prior[1] = float(len(training_data[training_data[:,0] == 1]))
    prior[2] = float(len(training_data[training_data[:,0] == 2]))
    prior[3] = float(len(training_data)) # total n observations
    print("priors = %s" % (prior))

    # build the histogram of heights for each class
    for height in training_data:
        height_hist[height[0]][int(height[1] / 10.0)] += 1.0

    # print("male counts :")
    # for h in height_hist[1]:
    #     print("[%.1f - %.1f[ cm : %d" % ((h * 10.0), (h * 10.0) + 10.0, height_hist[1][h]))

    # print("female counts :")
    # for h in height_hist[2]:
    #     print("[%.1f - %.1f[ cm : %d" % ((h * 10.0), (h * 10.0) + 10.0, height_hist[2][h]))

    # class conditional probabilities (on weight, using Parzen Window method)
    v = training_data[training_data[:,0] == 1][:,2]
    prob_weight_given_gender[1] = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(v.reshape(v.size, 1))

    v = training_data[training_data[:,0] == 2][:,2]
    prob_weight_given_gender[2] = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(v.reshape(v.size, 1))

    # plot weight distributions
    fig = plt.figure(figsize = (18, 5))
    ax1 = fig.add_subplot(131)

    x = np.arange(min(training_data[:,2]), max(training_data[:,2]), 0.1)
    ax1.plot(x, np.exp(prob_weight_given_gender[1].score_samples(x.reshape(x.size, 1))), color = 'blue', linewidth = 1.5, label = 'M')
    ax1.plot(x, np.exp(prob_weight_given_gender[2].score_samples(x.reshape(x.size, 1))), color = 'red', linewidth = 1.5, label = 'F')

    ax1.set_title('P(W|C)')

    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)

    ax1.legend(fontsize = 12, ncol = 1, loc='upper right')
    ax1.set_xlabel("weight (kg)")
    x = np.arange(int(min(training_data[:,2]) / 10.0) * 10.0, int(max(training_data[:,2]) / 10.0) * 11.0, 10.0)
    ax1.set_xlim(x[0] - 5, x[-1] + 5)
    ax1.set_xticks(x)
    ax1.set_ylabel("p(weight)")

    # plot height distributions
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    x_ticks = []
    for i in np.arange(int(min(training_data[:,1]) / 10.0), int(max(training_data[:,1]) / 10.0) + 1, 1):
        
        x_ticks.append((float(i) * 10.0))

        if i in height_hist[1]:
            print("P(h = [%.2f-%.2f]|M) = %.6f" % (float(i) * 10.0, float(i + 1) * 10.0, height_hist[1][i] / (10.0 * prior[1])))
            ax2.bar((float(i) * 10.0), height_hist[1][i] / (10.0 * prior[1]), alpha = 0.55, width = 10.0, color = 'blue')

        if i in height_hist[2]:
            print("P(h = [%.2f-%.2f]|F) = %.6f" % (float(i) * 10.0, float(i + 1) * 10.0, height_hist[2][i] / (10.0 * prior[2])))
            ax3.bar((float(i) * 10.0), height_hist[2][i] / (10.0 * prior[2]), alpha = 0.55, width = 10.0, color = 'red')

    x_ticks.append(x_ticks[-1] + 10.0)

    ax2.xaxis.grid(True)
    ax2.yaxis.grid(True)
    ax3.xaxis.grid(True)
    ax3.yaxis.grid(True)

    ax2.set_title('P(H|M)')
    ax3.set_title('P(H|F)')

    ax2.set_xlabel("height (cm)")
    ax3.set_xlabel("height (cm)")

    ax2.set_xlim(x_ticks[0] - 5, x_ticks[-1] + 5)
    ax3.set_xlim(x_ticks[0] - 5, x_ticks[-1] + 5)

    ax2.set_xticks(x_ticks)
    ax3.set_xticks(x_ticks)

    ax2.set_ylabel("p(height)")
    ax3.set_ylabel("p(height)")

    fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.3, hspace = None)
    plt.savefig("ex2-weight-distributions.pdf", bbox_inches = 'tight', format = 'pdf')

    return prior, height_hist, prob_weight_given_gender

def prob_weight(w, gender, prob_weight_given_gender):
    # class-conditional prob P(w|G)
    print("P(w = %.3f|%d) = %.5f" % (w, gender, np.exp(prob_weight_given_gender[gender].score_samples(w))))
    return np.exp(prob_weight_given_gender[gender].score_samples(w))

def prob_height(h, gender, height_hist, prior):
    # find bind h belongs to
    bin = int(h / 10.0)
    # if the bin isn't in the gender's histogram, P(G|X = x) is 0
    if bin not in height_hist[gender]:
        return 0.0
    else:
        # class-conditional prob P(h|G)
        print("P(h = %.3f|%d) = %.5f" % (h, gender, (height_hist[gender][bin] / (10.0 * prior[gender]))))
        return (height_hist[gender][bin] / (10.0 * prior[gender]))

def classify(training_data, test_data):

    prior, height_hist, prob_weight_given_gender = train(training_data)

    y = []

    for td in test_data:
        
        prob_male   = (prob_height(td[0], 1, height_hist, prior) * prob_weight(td[1], 1, prob_weight_given_gender) * (prior[1] / prior[3]))
        prob_female = (prob_height(td[0], 2, height_hist, prior) * prob_weight(td[1], 2, prob_weight_given_gender) * (prior[2] / prior[3]))

        print("\tP(M|w = %.1f, h = %.1f) ~ %.6f" % (td[1], td[0], prob_male))
        print("\tP(F|w = %.1f, h = %.1f) ~ %.6f" % (td[1], td[0], prob_female))
        print("\n")

        if (prob_male > prob_female):
            y.append(1)
        else:
            y.append(2)

    return y

if __name__ == "__main__":

    # read training data
    training_data = np.loadtxt(open("heightWeightData.txt", "rb"), delimiter=",", skiprows = 0)

    test_data = [[165, 80], [181, 65], [161, 57], [181, 77]]
    test_data = np.array(test_data)
    print(test_data)
    y = classify(training_data, test_data)
    print(y)

    # 2.c)
    prior, height_hist, prob_weight_given_gender = train(training_data)
    x = [165, 80]
    print("P(h = %.3f|M) = %.10f" % (x[0], (height_hist[1][int(x[0] / 10.0)] / (10.0 * prior[1]))))
    print("P(w = %.3f|M) = %.10f" % (x[1], np.exp(prob_weight_given_gender[1].score_samples(x[1]))))
    print("P([%.3f, %.3f]|M) : %.10f" % (x[0], x[1], (height_hist[1][int(x[0] / 10.0)] / prior[1]) * np.exp(prob_weight_given_gender[1].score_samples(x[1]))))
