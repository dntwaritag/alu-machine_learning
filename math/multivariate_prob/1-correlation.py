#!/usr/bin/env python3
'''
    Function that calculates the correlation matrix
    from a given covariance matrix.
'''


import numpy as np


def correlation(C):
    '''
    Arguments:
        C: A numpy array of shape (d, d) representing
        the covariance matrix.

    Returns:
        A numpy array of shape (d, d) containing
        the computed correlation matrix.
    '''
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    corr = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            corr[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])

    return corr
