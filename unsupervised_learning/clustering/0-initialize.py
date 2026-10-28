#!/usr/bin/env python3
"""Initialize K-means centroids."""
import numpy as np


def initialize(X, k):
    """Sample k centroids uniformly within the dataset bounds."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or 0 in X.shape):
        return None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None
    return np.random.uniform(X.min(axis=0), X.max(axis=0),
                             (k, X.shape[1]))
