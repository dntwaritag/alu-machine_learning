#!/usr/bin/env python3
'''Comment'''

import numpy as np


class NeuralNetwork:
    '''Private instance attributes:
    W1: The weights vector for the hidden layer. Upon instantiation,
    it should be initialized using a random normal distribution.
    b1: The bias for the hidden layer. Upon instantiation,
    it should be initialized with 0’s.
    A1: The activated output for the hidden layer.
    Upon instantiation, it should be initialized to 0.
    W2: The weights vector for the output neuron. Upon instantiation,
    it should be initialized using a random normal distribution.
    b2: The bias for the output neuron. Upon instantiation,
    it should be initialized to 0.
    A2: The activated output for the output neuron (prediction).
    Upon instantiation, it should be initialized to 0.'''

    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer.')

        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''getter'''
        return self.__W1

    @property
    def b1(self):
        '''getter'''
        return self.__b1

    @property
    def A1(self):
        '''getter'''
        return self.__A1

    @property
    def W2(self):
        '''getter'''
        return self.__W2

    @property
    def b2(self):
        '''getter'''
        return self.__b2

    @property
    def A2(self):
        '''getter'''
        return self.__A2

    def forward_prop(self, X):
        '''comment'''
        self.__A1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-self.__A1))
        self.__A2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-self.__A2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''
            Calculates the cost of the model using logistic regression
        '''
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) +
                                  (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluate the neuron's predictions'''
        A1, A2 = self.forward_prop(X)  # Get the activated output (predictions)
        # Convert probabilities (A) to binary predictions (0 or 1)
        predictions = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)  # Calculate the cost based on predictions
        return predictions, cost
