#!/usr/bin/env python3
""" train"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """ training operation"""
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=alpha)

    # Create the training operation
    train_op = optimizer.minimize(loss)

    return train_op
