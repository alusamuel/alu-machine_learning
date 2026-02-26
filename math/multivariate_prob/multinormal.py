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

    def pdf(self, x):
        """Calculates the PDF at a data point x."""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # PDF formula:
        # f(x) = 1 / sqrt((2π)^d det(Σ)) * exp(-1/2 (x-μ)^T Σ^-1 (x-μ))
        x_mu = x - self.mean

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        exponent = -0.5 * (x_mu.T @ inv @ x_mu)
        num = np.exp(exponent)[0, 0]

        denom = ((2 * np.pi) ** (d / 2)) * (det ** 0.5)

        return num / denom
