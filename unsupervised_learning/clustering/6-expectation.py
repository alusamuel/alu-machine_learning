#!/usr/bin/env python3
"""Expectation step for a Gaussian mixture model."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Return component responsibilities and total log likelihood."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or 0 in X.shape):
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1 or not pi.size:
        return None, None
    k, d = pi.size, X.shape[1]
    if (not isinstance(m, np.ndarray) or m.shape != (k, d)
            or not isinstance(S, np.ndarray) or S.shape != (k, d, d)
            or np.any(pi < 0) or not np.isclose(pi.sum(), 1)):
        return None, None
    weighted = np.empty((k, X.shape[0]))
    for j in range(k):
        P = pdf(X, m[j], S[j])
        if P is None:
            return None, None
        weighted[j] = pi[j] * P
    total = weighted.sum(axis=0)
    return weighted / total, np.log(total).sum()
