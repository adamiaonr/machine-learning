import numpy as np

# def gauss_bad(data, mean, std_dev):
#     return (1.0 / ((np.sqrt(2.0 * np.pi) * std_dev))) * np.exp(-((xx - mean) ** 2) / (2.0 * (std_dev ** 2)))

def gauss(data, mean, std_dev, w = 0.001):

    # since this is a hmm w/ continuous outputs, we can't directly 
    # use the pdf formulas.

    # here we aprox. the integral of the pdf in intervals of 0.1 
    # of the gaussian, using a trapezoid method. the width of the trapezoids is w.
    res = []
    for d in data:
        xx = np.arange(d, d + 0.1, w)
        xx = (1.0 / ((np.sqrt(2.0 * np.pi) * std_dev))) * np.exp(-((xx - mean) ** 2) / (2.0 * (std_dev ** 2)))
        xx = xx * w

        res.append(np.sum(xx))

    return np.array(res)

def unif(data, l, h):
    res = []
    for d in data:
        res.append((1.0 / float(h - l)) if (d >= l and d <= h) else 0.0)

    return np.array(res)

if __name__ == "__main__":

    # observations
    xx = np.array([0.7, 0.7, 0.1, 0.2, 0.3, 0.6, 0.2, 0.3, -0.1, 0.2])

    # hmm model:
    #   - transition probabilities (between hidden states)
    T = np.array([  [0.90, 0.10], 
                    [0.10, 0.90]])
    #   - start probabilities
    pi0 = np.array([0.50, 0.50])[:,None]
    #   - emission probabilities
    Px = np.array([gauss(xx, 0.0, 0.2), unif(xx, 0.0, 1.0) * 0.10])

    # forward method - slide 31
    alpha = pi0 * Px[:, [0]]
    for t in range(1, np.size(xx)):
        alpha = Px[:, [t]] * np.dot(T.T, alpha)

    # prob of the sequence of observations
    seq_prob = np.sum(alpha, 0)
    print("hw5::ex4 : prob of seq. : %s" % (str(seq_prob)))
