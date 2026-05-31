#!/usr/bin/env python3
"""
Create an RMSProp optimizer op in tensorflow.
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation using RMSProp optimization.

    Parameters:
    - loss: loss tensor
    - alpha: learning rate
    - beta2: RMSProp weight
    - epsilon: small value to avoid division by zero

    Returns:
    - train_op: optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
    )
    train_op = optimizer.minimize(loss)
    return train_op
