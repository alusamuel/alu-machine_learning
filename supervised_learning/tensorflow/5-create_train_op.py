#!/usr/bin/env python3
"""Defines a function that creates the training operation."""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Create the training operation.

    Args:
        loss: network loss.
        alpha: learning rate.

    Returns:
        Training operation.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
