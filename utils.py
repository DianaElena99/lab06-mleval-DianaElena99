import numpy as np
from math import exp


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

if __name__ == '__main__':
    print(softmax([2.0, 1.0, 0.1]))