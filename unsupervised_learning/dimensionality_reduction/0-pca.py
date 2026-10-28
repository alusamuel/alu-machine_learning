#!/usr/bin/env python3
"""Principal component weights for a centered dataset."""
import numpy as np


def pca(X, var=0.95):
    """Return axes retaining the requested singular-value fraction."""
    _, singular_values, Vh = np.linalg.svd(X, full_matrices=False)
    cumulative = np.cumsum(singular_values) / np.sum(singular_values)
    ndim = np.searchsorted(cumulative, var) + 1
    return Vh[:ndim].T
