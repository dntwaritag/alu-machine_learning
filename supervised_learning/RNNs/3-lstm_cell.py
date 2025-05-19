#!/usr/bin/env python3
"""
    Script that defines a class LSTMCell
    that represents an LSTM unit
"""


import numpy as np


class LSTMCell:
    """
    Class LSTMCell that represents an LSTM unit

    parameters:
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
    """

    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """
        Performs the sigmoid function
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def softmax(self, x):
        """
        Performs the softmax function
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def forward(self, h_prev, c_prev, x_t):
        """
        Function that performs forward propagation
        """

        concatenation = np.concatenate((h_prev, x_t), axis=1)
        ft = self.sigmoid(np.matmul(concatenation, self.Wf) + self.bf)
        ut = self.sigmoid(np.matmul(concatenation, self.Wu) + self.bu)
        ct = np.tanh(np.matmul(concatenation, self.Wc) + self.bc)
        c_next = ft * c_prev + ut * ct
        ot = self.sigmoid(np.matmul(concatenation, self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y
