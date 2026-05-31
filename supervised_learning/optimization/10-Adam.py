#!/usr/bin/env python3
"""
Create an Adam optimizer op in tensorflow.
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation using Adam optimization.

    Parameters:
    - loss: loss tensor
    - alpha: learning rate
    - beta1: first moment weight
    - beta2: second moment weight
    - epsilon: small value to avoid division by zero

    Returns:
    - train_op: optimization operation
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )
    train_op = optimizer.minimize(loss)
    return train_op