#!/usr/bin/env python3
"""One-hot decode labels"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    one_hot: numpy.ndarray of shape (classes, m)
    Returns: numpy.ndarray of shape (m,) with labels, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    # Check entries are non-negative before decoding.
    if not np.all(one_hot >= 0):
        return None

    labels = np.argmax(one_hot, axis=0)
    return labels
