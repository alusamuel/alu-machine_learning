#!/usr/bin/env python3
"""Deep neural network for binary classification (private attributes)"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        nx: number of input features
        layers: list of nodes in each layer of the network
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0 or not all(
            isinstance(n, int) and n > 0 for n in layers
        ):
            raise TypeError("layers must be a list of positive integers")

        # Private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # He initialization (one loop)
        for layer in range(1, self.__L + 1):
            if layer == 1:
                n_prev = nx
            else:
                n_prev = layers[layer - 2]
            n_curr = layers[layer - 1]

            self.__weights["W{}".format(layer)] = (
                np.random.randn(n_curr, n_prev) * np.sqrt(2 / n_prev)
            )
            self.__weights["b{}".format(layer)] = np.zeros((n_curr, 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Dictionary holding intermediary values"""
        return self.__cache

    @property
    def weights(self):
        """Dictionary holding weights and biases"""
        return self.__weights
