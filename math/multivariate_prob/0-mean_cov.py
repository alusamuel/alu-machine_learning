#!/usr/bin/env python3
"""Mean and Covariance module"""

import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Compute mean (shape: 1, d)
    mean = np.mean(X, axis=0, keepdims=True)

    # Center the data
    X_centered = X - mean

    # Compute covariance matrix (shape: d, d)
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
