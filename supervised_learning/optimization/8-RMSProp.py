#!/usr/bin/env python3
""" Training with RMSProp optimization
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ Creates the training operation for a neural network
    using RMSProp optimization.
    Args:
        loss: The loss of the network.
        alpha: The learning rate.
        beta2: The RMSProp decay factor (moving average decay).
        epsilon: A small number to avoid division by zero.
    Returns:
        The RMSProp optimization operation.
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    return optimizer.minimize(loss)
