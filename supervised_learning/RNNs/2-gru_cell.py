#!/usr/bin/env python3
'''
    Script that defines a class GRUCell
    GRUCell that represents a gated recurrent unit
'''


import numpy as np


class GRUCell:
    '''
        Class GRUCell that represents a gated recurrent unit

        parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
    '''

    def __init__(self, i, h, o):
        '''
            Class constructor
        '''

        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Performs the softmax function

        parameters:
            x: the value to perform softmax on to generate output of cell

        return:
            softmax of x
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def sigmoid(self, x):
        """
        Performs the sigmoid function

        parameters:
            x: the value to perform sigmoid on

        return:
            sigmoid of x
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, h_prev, x_t):
        '''
            Function that performs forward propagation

            parameters:
                h_prev: contains the previous hidden state
                x_t: contains the data input of the cell

            return:
                h_next: next hidden state
                y: output of the cell
        '''

        concatenation1 = np.concatenate((h_prev, x_t), axis=1)
        z_gate = self.sigmoid(np.matmul(concatenation1, self.Wz) + self.bz)
        r_gate = self.sigmoid(np.matmul(concatenation1, self.Wr) + self.br)

        concatenation2 = np.concatenate((r_gate * h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation2, self.Wh) + self.bh)
        h_next *= z_gate
        h_next += (1 - z_gate) * h_prev

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
