#!/usr/bin/env python3
""" Calculate Sensitivity """

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters:
    - confusion: is a confusion numpy.ndarray of shape (classes, classes) where
      row indices represent the correct labels and column indices represent the
      predicted labels

    Returns:
    A numpy.ndarray of shape (classes,) containing the
    sensitivity of each class.
    """
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    sensitivity = true_positives / (true_positives + false_negatives)
    return sensitivity
