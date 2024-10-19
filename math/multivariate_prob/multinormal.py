#!/usr/bin/env python3
"""
    A class MultiNormal to represent
    a multivariate normal (Gaussian) distribution.
"""

import numpy as np

class MultiNormal:
    """
    A class to model a multivariate normal distribution.
    """

    def __init__(self, data):
        """
        Initializes the MultiNormal instance with the given data.

        Args:
            data: A numpy array of shape (n, d) representing
                  the data set, where n is the number of dimensions
                  and d is the number of data points.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy array")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain more than one data point")

        self.data = data
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov

    def pdf(self, x):
        """
        Computes the probability density function (PDF) at a given point.

        Args:
            x: A numpy array of shape (d,) representing the point
               at which the PDF is to be evaluated. Here, d is the
               number of dimensions of the distribution.

        Returns:
            The value of the PDF evaluated at x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError(f"x must have the shape ({d}, 1)")
        test_d, one = x.shape
        if test_d != d or one != 1:
            raise ValueError(f"x must have the shape ({d}, 1)")

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        pdf = pdf[0][0]
        return pdf
