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
    
        # use higher precision for the internal computation
        cov = self.cov.astype(np.longdouble)
        mean = self.mean.astype(np.longdouble)
        x = x.astype(np.longdouble)
    
        x_mu = x - mean
    
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
    
        exponent = (-np.longdouble(0.5)) * (x_mu.T @ inv @ x_mu)
        num = np.exp(exponent)[0, 0]
    
        denom = np.sqrt(((np.longdouble(2.0) * np.pi) ** np.longdouble(d)) * det)
    
        return float(num / denom)
