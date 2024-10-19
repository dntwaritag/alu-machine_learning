#!/usr/bin/env python3

"""
Defines functions to calculate the intersection and marginal
probabilities of obtaining data given various hypothetical probabilities
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


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining the observed data
    with the various hypothetical probabilities.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy.ndarray containing the various
        hypothetical probabilities of developing severe side effects.
        Pr (numpy.ndarray): A 1D numpy.ndarray,
        containing the prior beliefs of P.

    Returns:
        numpy.ndarray: A 1D numpy.ndarray containing the intersection
        of obtaining x and n with each probability in P, respectively.

    Raises:
        ValueError: If n is not a positive integer,
        if x is not an integer that is greater than or equal to 0,
        if x is greater than n, if Pr does not sum to 1,
        or if any value in P or Pr is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray or
        if Pr is not a numpy.ndarray with the same shape as P.
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

    # Check if Pr is a numpy.ndarray with the same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Check if all values in P are in the range [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Check if all values in Pr are in the range [0, 1]
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Check if Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood using the likelihood function
    likelihoods = likelihood(x, n, P)

    # Calculate the intersection
    intersection = likelihoods * Pr

    return intersection


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy.ndarray containing the various
        hypothetical probabilities of developing severe side effects.
        Pr (numpy.ndarray): A 1D
        numpy.ndarray containing the prior beliefs of P.

    Returns:
        float: The marginal probability of obtaining x and n.

    Raises:
        ValueError: If n is not a positive integer,
        if x is not an integer that is greater than or equal to 0,
        if x is greater than n, if Pr does not sum to 1,
        or if any value in P or Pr is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray or
        if Pr is not a numpy.ndarray with the same shape as P.
    """
    # Calculate the intersection using the intersection function
    intersections = intersection(x, n, P, Pr)

    # Calculate the marginal probability by summing the intersections
    marginal_prob = np.sum(intersections)

    return marginal_prob


if __name__ == '__main__':
    P = np.linspace(0, 1, 21)  # [0.0, 0.05, 0.1, ..., 1.0]
    Pr = np.ones(21) / 21
    print(marginal(55, 100, P, Pr).round(12))
