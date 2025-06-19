#!/usr/bin/env python3
"""This module contains a function that finds the best number of clusters
for a GMM using the Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the
    Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= kmin:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    likelihoods = []
    bics = []
    best_k = None
    best_result = None
    best_bic = float('inf')

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol,
                                                 verbose)
        if pi is None:
            return None, None, None, None

        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)
        bic = p * np.log(n) - 2 * ll
        likelihoods.append(ll)
        bics.append(bic)

        if bic < best_bic:
            best_k = k
            best_result = (pi, m, S)
            best_bic = bic

    if best_k is None:
        return None, None, None, None

    return best_k, best_result, np.array(likelihoods), np.array(bics)
