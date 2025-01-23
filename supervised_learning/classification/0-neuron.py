#!/usr/bin/env python3
'''Neuron'''

import numpy as np

'''This class is a single neuron for binary classification'''


class Neuron:
    """
        Initialize a Neuron performing binary classification.

        Args:
            nx (int): The number of input features to the neuron.
        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
