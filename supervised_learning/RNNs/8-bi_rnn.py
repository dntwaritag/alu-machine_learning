#!/usr/bin/env python3
'''
Script that defines a function def bi_rnn(bi_cell, X, h_0, h_t):
that performs forward propagation for a bidirectional RNN:
'''


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_T):
    '''
    Function that performs forward propagation for a bidirectional RNN

    parameters:
        bi_cell: an instance of BidirectionalCell
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
        h_0: initial hidden state in the forward direction,
             numpy.ndarray of shape (m, h)
        h_T: initial hidden state in the backward direction,
             numpy.ndarray of shape (m, h)

    return:
        H: numpy.ndarray containing all of the concatenated hidden states
        Y: numpy.ndarray containing all of the outputs
    '''
    t, m, i = X.shape
    _, h = h_0.shape

    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    for step in range(t):
        if step == 0:
            h_next = h_0
        h_next = bi_cell.forward(h_next, X[step])
        H_forward[step] = h_next

    for step in range(t-1, -1, -1):
        if step == t-1:
            h_prev = h_T
        h_prev = bi_cell.backward(h_prev, X[step])
        H_backward[step] = h_prev

    H = np.concatenate((H_forward, H_backward), axis=2)
    Y = bi_cell.output(H)

    return H, Y
