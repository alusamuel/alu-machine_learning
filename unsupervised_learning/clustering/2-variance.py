#!/usr/bin/env python3
"""Total intra-cluster variance."""
import numpy as np


def variance(X, C):
    """Sum squared distances from samples to their nearest centroid."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or 0 in X.shape):
        return None
    if (not isinstance(C, np.ndarray) or C.ndim != 2
            or 0 in C.shape or C.shape[1] != X.shape[1]):
        return None
    return np.sum(np.min(np.sum((X[:, None] - C) ** 2, axis=2), axis=1))
