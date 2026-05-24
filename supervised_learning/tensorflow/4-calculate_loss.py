#!/usr/bin/env python3
"""Defines a function that calculates softmax cross-entropy loss."""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate the loss.

    Args:
        y: true labels placeholder.
        y_pred: predicted logits tensor.

    Returns:
        Tensor containing the loss.
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)