#!/usr/bin/env python3
'''This class is a single neuron for binary classification'''

import numpy as np


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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''Getter for Weights'''
        return self.__W

    @property
    def b(self):
        '''Getter for Bias'''
        return self.__b

    @property
    def A(self):
        '''Getter for Activation function'''
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m), where
            nx is the number of input features
            and m is the number of examples.

        Returns:
            numpy.ndarray: The activated output of the neuron (A).
        """
        # Linear combination Z = W.X + b
        Z = np.dot(self.__W, X) + self.__b

        # Sigmoid activation function A = 1 / (1 + e^(-Z))
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
