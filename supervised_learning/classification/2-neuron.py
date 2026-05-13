#!/usr/bin/env python3
"""Single neuron performing binary classification (forward propagation)"""
import numpy as np


class Neuron:
    """Defines a single neuron for binary classification"""

    def __init__(self, nx):
        """
        Initialize a neuron.

        nx: number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neuron.

        X: numpy.ndarray of shape (nx, m) with input data
        """
        # Linear part: Z = W.X + b
        Z = np.matmul(self.__W, X) + self.__b
        # Sigmoid activation
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
