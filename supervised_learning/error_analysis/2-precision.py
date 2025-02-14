#!/usr/bin/env python3
""" Calculate Precision """

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Parameters:
    - confusion: is a confusion numpy.ndarray of shape (classes, classes) where
      row indices represent the correct labels and column indices represent the
      predicted labels

    Returns:
    A numpy.ndarray of shape (classes,) containing the precision of each class.
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    precision = true_positives / (true_positives + false_positives)
    return precision
