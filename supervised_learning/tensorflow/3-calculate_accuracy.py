#!/usr/bin/env python3
"""Defines a function that calculates prediction accuracy."""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate accuracy.

    Args:
        y: true labels placeholder.
        y_pred: predicted logits tensor.

    Returns:
        Tensor containing decimal accuracy.
    """
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
