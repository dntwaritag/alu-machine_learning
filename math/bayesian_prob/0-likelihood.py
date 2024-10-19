#!/usr/bin/env python3

"""
Defines a function likelihood that calculates the likelihood
of obtaining data given various hypothetical probabilities
of developing severe side effects.
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the observed data
    given various hypothetical probabilities of developing severe side effects.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy.ndarray containing the various
        hypothetical probabilities of developing severe side effects.

    Returns:
        numpy.ndarray: A 1D numpy.ndarray containing the likelihood
        of obtaining the data,x and n, for each probability in P, respectively.

    Raises:
        ValueError: If n is not a positive integer,
        if x is not an integer that is greater than or equal to 0,
        if x is greater than n,or if any value in P is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray.
    """
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is an integer and greater than or equal to 0
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    # Check if x is greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in P are in the range [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the combination using np.math.factorial
    fact_coefficient = np.math.factorial(
        n) / (np.math.factorial(x) * np.math.factorial(n - x))

    # Calculate likelihoods
    likelihoods = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods


if __name__ == '__main__':
    P = np.linspace(0, 1, 21)  # [0.0, 0.05, 0.1, ..., 1.0]
    print(likelihood(55, 100, P).round(12))