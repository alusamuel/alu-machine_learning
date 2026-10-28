#!/usr/bin/env python3
"""Multivariate Gaussian probability density."""
import numpy as np


def pdf(X, m, S):
    """Evaluate a Gaussian density, flooring probabilities at 1e-300."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or 0 in X.shape):
        return None
    d = X.shape[1]
    if (not isinstance(m, np.ndarray) or m.shape != (d,)
            or not isinstance(S, np.ndarray) or S.shape != (d, d)
            or not np.allclose(S, S.T)):
        return None
    try:
        np.linalg.cholesky(S)
        sign, logdet = np.linalg.slogdet(S)
        diff = X - m
        quadratic = np.sum(diff * np.linalg.solve(S, diff.T).T, axis=1)
    except np.linalg.LinAlgError:
        return None
    log_p = -0.5 * (d * np.log(2 * np.pi) + logdet + quadratic)
    return np.maximum(np.exp(log_p), 1e-300)
