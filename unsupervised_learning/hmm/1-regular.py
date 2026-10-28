#!/usr/bin/env python3
"""Stationary distributions of regular Markov chains."""
import numpy as np


def regular(P):
    """Return the stationary distribution if P is a regular chain."""
    if (not isinstance(P, np.ndarray) or P.ndim != 2
            or P.shape[0] == 0 or P.shape[0] != P.shape[1]
            or not np.all(np.isfinite(P)) or np.any(P < 0)
            or not np.allclose(P.sum(axis=1), 1)):
        return None
    n = P.shape[0]
    # A primitive n-by-n matrix has exponent at most (n - 1)^2 + 1.
    reachable = np.linalg.matrix_power(P > 0, (n - 1) ** 2 + 1)
    if not np.all(reachable):
        return None
    equations = np.vstack((P.T - np.eye(n), np.ones((1, n))))
    targets = np.append(np.zeros(n), 1)
    stationary = np.linalg.lstsq(equations, targets, rcond=None)[0]
    stationary = np.maximum(stationary, 0)
    return (stationary / stationary.sum()).reshape(1, n)
