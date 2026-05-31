#!/usr/bin/env python3
"""
Create a momentum optimizer op in tensorflow.
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network using momentum.

    Parameters:
    - loss: loss tensor
    - alpha: learning rate
    - beta1: momentum weight

    Returns:
    - train_op: optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha,
                                           momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op