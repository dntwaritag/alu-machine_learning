#!/usr/bin/env python3
"""
Defines a function to calculate the definiteness of a matrix.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): Matrix to be evaluated.

    Returns:
        str: Definiteness of the matrix, or None if not symmetric,
             or raises TypeError for incorrect types.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.allclose(matrix, matrix.T):
        return None  # Return None for non-symmetric matrix

    eigenvalues = np.linalg.eigvals(matrix)
    all_positive = np.all(eigenvalues > 0)
    all_nonnegative = np.all(eigenvalues >= 0) and not all_positive
    all_negative = np.all(eigenvalues < 0)
    all_nonpositive = np.all(eigenvalues <= 0) and not all_negative

    if all_positive:
        return "Positive definite"
    if all_nonnegative:
        return "Positive semi-definite"
    if all_negative:
        return "Negative definite"
    if all_nonpositive:
        return "Negative semi-definite"
    return "Indefinite"
