#!/usr/bin/env python3
"""MultiNormal module"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        Initializes the MultiNormal instance

        Parameters:
        data (numpy.ndarray): shape (d, n)
            d = number of dimensions
            n = number of data points
        """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean (shape: d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data
        data_centered = data - self.mean

        # Covariance matrix (shape: d, d)
        self.cov = (data_centered @ data_centered.T) / (n - 1)
