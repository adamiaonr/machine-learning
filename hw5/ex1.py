import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from collections import defaultdict

def gaussian(data, mean, std_dev):
    return (1.0 / ((np.sqrt(2.0 * np.pi) * std_dev))) * np.exp(-((data - mean) ** 2) / (2.0 * (std_dev ** 2)))

# expectation (E) step : update responsibilities, i.e.
# posterior probabilities p(k|x), the probability of 
# mix component k producing x 
def e_step(data, mean, covar, mix_coef):
    R = np.array([(gaussian(x, mean, np.sqrt(covar)) * mix_coef) for x in data])
    return (R / np.sum(R, 1)[:, None])

# maximization (M) step : update parameters, 
# using updated responsibilities from E step
def m_step(data, R):

    # sum of responsibilities per component 
    # (for all data points)
    N_k = np.sum(R, 0)

    # mean    
    mean = (1.0 / N_k) * np.dot(R.T, data)

    # covariance (variance in 1d)
    covar = np.zeros(len(mean))
    for i, x in enumerate(data):
        covar += ((x - mean) * (x - mean) * R[i,:])
    covar = (1.0 / N_k) * covar

    # mix coefficients
    mix_coef = N_k / len(data)

    return mean, covar, mix_coef

def log_likelihood(data, mean, covar, mix_coef):
    return np.sum([np.log(np.sum(gaussian(x, mean, np.sqrt(covar)) * mix_coef)) for x in data])

if __name__ == "__main__":

    # data = np.array([1.0, 10.0, 20.0])
    # responsibilites matrix
    # R = np.array([  [1.0, 0.0], 
    #                 [0.4, 0.6],
    #                 [0.0, 1.0]])

    # m_step(R, data)

    # for reproducible random results
    np.random.seed(110)

    # draw random data points from 2 different 1d gaussians
    #   - red   : [mean : -2.0, std dev : 0.8]
    #   - green : [mean : 0.0, std dev : 0.5]
    #   - blue  : [mean : 3.0, std dev : 2.0]
    ground_truth = {'red' : [-5.0, 1.5], 'green' : [0.0, 0.5], 'blue' : [5.0, 2.0]}

    # generate data accordint to mix of gaussians
    fig = plt.figure(figsize = (6, 5))

    # t-sne
    ax = fig.add_subplot(111)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    data = defaultdict()
    data['total'] = np.array([])
    for dist in ground_truth:
        data[dist] = np.random.normal(ground_truth[dist][0], ground_truth[dist][1], size = 40)
        data['total'] = np.sort(np.concatenate((data['total'], data[dist])))

        ax.scatter(data[dist], np.zeros(len(data[dist])), color = dist, marker = 'o', s = 10)

    # run expectation maximization over data['total']. assume a mix of k gaussians.
    k = 3
    
    # initialize mean, covariances (1d) and mix coefficients
    mean = np.array([-10.0, 3.0, 10.0])
    covar = np.array([5.0, 5.0 , 5.0])
    mix_coef = np.ones(k) * (1.0 / float(k))

    # evaluate log likelihood, given the parameters
    ll = log_likelihood(data['total'], mean, covar, mix_coef)

    colors = ['red', 'green', 'blue']
    for i in xrange(500):

        if ((i % 100) == 0):
            for k, c in enumerate(colors):
                x = np.arange(mean[k] - 2.0 * np.sqrt(covar[k]), mean[k] + 2.0 * np.sqrt(covar[k]), 0.01)
                ax.plot(x, gaussian(x, mean[k], np.sqrt(covar[k])), color = c, linewidth = 1.0, alpha = (0.10 * ((i / 100) + 1)))

        R = e_step(data['total'], mean, covar, mix_coef)
        mean, covar, mix_coef = m_step(data['total'], R)

    fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.3, hspace = None)
    plt.savefig("ex1.pdf", bbox_inches = 'tight', format = 'pdf')
