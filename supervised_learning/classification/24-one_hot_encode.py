#!/usr/bin/env python3
"""One-hot encode labels"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Y: numpy.ndarray of shape (m,) containing class labels
    classes: number of classes
    Returns: one-hot encoding of shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None
    if np.any((Y < 0) | (Y >= classes)):
        return None

    one_hot = np.zeros((classes, Y.shape[0]), dtype=float)
    one_hot[Y, np.arange(Y.shape[0])] = 1.
    return one_hot
