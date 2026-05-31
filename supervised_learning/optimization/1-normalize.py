#!/usr/bin/env python3
"""
Module for normalizing (standardizing) a matrix using given mean and std.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
    - X: np.ndarray of shape (d, nx)
    - m: np.ndarray of shape (nx,)
        Mean of features of X
    - s: np.ndarray of shape (nx,)
        Standard deviation of features of X

    Returns:
    - X_norm: np.ndarray
        Normalized X
    """
    X_norm = (X - m) / s
    return X_norm