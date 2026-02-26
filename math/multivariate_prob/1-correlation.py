#!/usr/bin/env python3
"""Correlation module"""

import numpy as np


def correlation(C):
    """Calculates a correlation matrix from a covariance matrix"""

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Standard deviations (sqrt of diagonal)
    std = np.sqrt(np.diag(C))

    # Outer product of std deviations
    std_outer = np.outer(std, std)

    # Correlation matrix
    corr = C / std_outer

    return corr
