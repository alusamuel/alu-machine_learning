#!/usr/bin/env python3
"""Initialize a Gaussian mixture model."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Use K-means means, equal priors, and identity covariances."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or 0 in X.shape):
        return None, None, None
    if type(k) is not int or not 0 < k <= X.shape[0]:
        return None, None, None
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    return np.full(k, 1 / k), m, np.tile(np.eye(X.shape[1]), (k, 1, 1))
