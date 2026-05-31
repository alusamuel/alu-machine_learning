#!/usr/bin/env python3
"""
Module for computing normalization (standardization) constants.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Parameters:
    - X: np.ndarray of shape (m, nx)
        m is the number of data points
        nx is the number of features

    Returns:
    - m: np.ndarray of shape (nx,)
        Mean of each feature
    - s: np.ndarray of shape (nx,)
        Standard deviation of each feature
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s