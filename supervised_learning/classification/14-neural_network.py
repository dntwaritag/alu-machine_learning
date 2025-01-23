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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''Perform one pass of gradient descent to update weights and bias'''
        m = Y.shape[1]  # Number of examples
        dz2 = A2 - Y
        dw2 = (1 / m) * np.dot(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 -= (alpha * dw2)
        self.__b2 -= (alpha * db2)
        self.__W1 -= (alpha * dw1)
        self.__b1 -= (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''This class trains the model
        Args:
            X is a numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
            Y is a numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
            iterations is the number of iterations to train over
            alpha is the learning rate
        '''
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        costs = []
        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            costs.append(cost)
            self.gradient_descent(X, Y, A1, A2, alpha)

        # evaluate the gradient descent
        evaluation = self.evaluate(X, Y)
        return evaluation
