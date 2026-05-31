#!/usr/bin/env python3
"""
Module for shuffling data matrices X and Y in the same way.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
    - X: np.ndarray of shape (m, nx)
    - Y: np.ndarray of shape (m, ny)

    Returns:
    - X_shuffled, Y_shuffled
    """
    m = X.shape[0]
    perm = np.random.permutation(m)
    return X[perm], Y[perm]
