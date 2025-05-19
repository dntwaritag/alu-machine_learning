#!/usr/bin/env python3
"""
    Script that defines a class
    BidirectionalCell
"""


import numpy as np


class BidirectionalCell:
    """
    Class that represents a bidirectional cell of an RNN

    class constructor
    """

    def __init__(self, i, h, o):
        """
        Class constructor
        """

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=((2 * h), o))

    def forward(self, h_prev, x_t):
        """
        Function that performs forward propagation
        """

        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Function that performs backward propagation

        parameters:
            h_next: next hidden state
            x_t: data

        return:
            h_prev: previous hidden state
        """

        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)

        return h_prev

    def softmax(self, x):
        '''
            Performs the softmax function
        '''
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def output(self, H):
        '''
            Function that returns all outputs from the bidirectional cell

            parameters:
                H: output of the forward cell

            return:
                y: output of the bidirectional cell
        '''
        t, m, h = H.shape

        Y = []

        for step in range(t):
            y = self.softmax(np.matmul(H[step], self.Wy) + self.by)
            Y.append(y)

        return np.array(Y)
