#!/usr/bin/env python3
'''Comment'''

import numpy as np


class NeuralNetwork:
    '''Public instance attributes:
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
            raise ValueError('nodes must be a positive integer')

        self.nx = nx
        self.nodes = nodes
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
