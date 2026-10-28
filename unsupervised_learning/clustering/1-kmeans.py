#!/usr/bin/env python3
"""K-means clustering using NumPy."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Return centroids and nearest-centroid labels, or a failure pair."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or 0 in X.shape):
        return None, None
    if (type(k) is not int or not 0 < k <= X.shape[0]
            or type(iterations) is not int or iterations <= 0):
        return None, None
    low, high = X.min(axis=0), X.max(axis=0)
    C = np.random.uniform(low, high, (k, X.shape[1]))
    for _ in range(iterations):
        clss = np.argmin(np.sum((X[:, None] - C) ** 2, axis=2), axis=1)
        updated = C.copy()
        for j in range(k):
            points = X[clss == j]
            if points.shape[0]:
                updated[j] = points.mean(axis=0)
            else:
                updated[j] = np.random.uniform(low, high, X.shape[1])
        if np.array_equal(C, updated):
            return C, clss
        C = updated
    clss = np.argmin(np.sum((X[:, None] - C) ** 2, axis=2), axis=1)
    return C, clss
