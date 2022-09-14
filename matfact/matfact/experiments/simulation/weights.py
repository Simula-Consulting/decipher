import numpy as np


def data_weights(X):
    """Example of a weight matrix used to focus on minority samples
    in the discrepancy term.
    """

    W = np.zeros_like(X)
    W[X == 1] = 1
    W[X == 2] = 2
    W[X == 3] = 3
    W[X == 4] = 4

    return W
