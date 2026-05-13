#!/usr/bin/env python3
"""Single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Defines a single neuron for binary classification"""

    def __init__(self, nx):
        """
        Initialize a neuron.

        nx: number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights: shape (1, nx), random normal
        self.W = np.random.randn(1, nx)
        # Bias: scalar 0
        self.b = 0
        # Activated output: scalar 0
        self.A = 0
