#!/usr/bin/env python3
"""Deep neural network for binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        nx: number of input features
        layers: list of nodes in each layer
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        # Public attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # He initialization for each layer (one loop allowed)
        for layer in range(1, self.L + 1):
            if layer == 1:
                n_prev = nx
            else:
                n_prev = layers[layer - 2]
            n_curr = layers[layer - 1]

            # He et al. initialization: N(0, sqrt(2 / n_prev))
            self.weights["W{}".format(layer)] = (
                np.random.randn(n_curr, n_prev) * np.sqrt(2 / n_prev)
            )
            self.weights["b{}".format(layer)] = np.zeros((n_curr, 1))
