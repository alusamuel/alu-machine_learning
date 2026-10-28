#!/usr/bin/env python3
"""Maximization step for a Gaussian mixture model."""
import numpy as np


def maximization(X, g):
    """Update priors, means, and covariances from responsibilities."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or 0 in X.shape):
        return None, None, None
    n, d = X.shape
    if (not isinstance(g, np.ndarray) or g.ndim != 2
            or g.shape[1] != n or g.shape[0] == 0
            or not np.allclose(g.sum(axis=0), 1)):
        return None, None, None
    mass = g.sum(axis=1)
    if np.any(mass == 0):
        return None, None, None
    pi = mass / n
    m = g @ X / mass[:, None]
    S = np.empty((g.shape[0], d, d))
    for j in range(g.shape[0]):
        diff = X - m[j]
        S[j] = (diff.T * g[j]) @ diff / mass[j]
    return pi, m, S
