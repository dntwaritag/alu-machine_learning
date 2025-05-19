#!/usr/bin/env python3
'''
    Script that defines a BidirectionalCell
    that represents a bidirectional cell of an RNN
'''


import numpy as np


class BidirectionalCell:
    '''
        Class that represents a bidirectional cell of an RNN

        class constructor
    '''

    def __init__(self, i, h, o):
        '''
            Class constructor
        '''

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=((2 * h), o))

    def forward(self, h_prev, x_t):
        '''
            Function that performs forward propagation
        '''

        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next
