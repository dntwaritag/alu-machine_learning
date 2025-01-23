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

    def cost(self, Y, A):
        """
            Calculates the cost of the model using logistic regression.

            Args:
                Y (numpy.ndarray): Correct labels (1, m) for the input data.
                A (numpy.ndarray): Activated output (1, m) for each example.

            Returns:
                float: The cost (cross-entropy loss).
            """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) +
                                  (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''This evaluates the Neuron
            Args:
                X (numpy.ndarray): Input data of shape (nx, m), where
            nx is the number of input features
            and m is the number of examples
                Y (numpy.ndarray): Correct labels (1, m) for the input data.

            Returns:
                Prdiction output of y converted to probabilties between 0 and 1
                cost between the lables
        '''
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''This functiong calculates the gradient descent of the neuron
        Args:
            X (numpy.ndarray): Input data of shape (nx, m), where
            nx is the number of input features
            and m is the number of examples
            Y (numpy.ndarray): Correct labels (1, m) for the input data.
            A (numpy.ndarray): Activated output (1, m) for each example.
            alpha is the learning rate of the gradient descent
        '''
        m = Y.shape[1]
        error = A - Y
        dW = (1 / m) * (np.dot(error, np.transpose(X)))
        db = (1 / m) * (np.sum(error))
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)
