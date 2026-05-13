#!/usr/bin/env python3
"""Deep neural network for binary classification (forward propagation)"""
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

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # He initialization
        for layer in range(1, self.__L + 1):
            if not isinstance(layers[layer - 1], int) or layers[layer - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
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

    def forward_prop(self, X):
        """
        Calculates forward propagation of the deep neural network.

        X: numpy.ndarray of shape (nx, m)
        Returns: output of the network, and the cache
        """
        # Save input as A0
        self.__cache["A0"] = X

        # Loop through layers 1..L
        for layer in range(1, self.__L + 1):
            Wl = self.__weights["W{}".format(layer)]
            bl = self.__weights["b{}".format(layer)]
            Al_prev = self.__cache["A{}".format(layer - 1)]

            Zl = np.matmul(Wl, Al_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))  # sigmoid

            self.__cache["A{}".format(layer)] = Al

        # Final output is AL
        return self.__cache["A{}".format(self.__L)], self.__cache
