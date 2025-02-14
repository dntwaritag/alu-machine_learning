#!/usr/bin/env python3
""" Calculate Specificity """

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Parameters:
    - confusion: is a confusion numpy.ndarray of shape (classes, classes) where
      row indices represent the correct labels and column indices represent the
      predicted labels

    Returns:
    A numpy.ndarray of shape (classes,) containing the
    specificity of each class.
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    true_negatives = np.sum(confusion) - \
        (false_positives + false_negatives + true_positives)

    specificity = true_negatives / (true_negatives + false_positives)
    return specificity
