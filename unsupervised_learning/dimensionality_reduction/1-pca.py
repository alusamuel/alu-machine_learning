#!/usr/bin/env python3
"""Project a dataset onto a fixed number of principal components."""
import numpy as np


def pca(X, ndim):
    """Center X and project it onto the first ndim principal axes."""
    centered = X - np.mean(X, axis=0)
    _, _, Vh = np.linalg.svd(centered, full_matrices=False)
    return np.matmul(centered, Vh[:ndim].T)
