#!/usr/bin/env python3
"""Deep neural network for binary classification (with evaluate)"""
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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # He initialization
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

    def forward_prop(self, X):
        """
        Calculates forward propagation of the deep neural network.

        X: numpy.ndarray of shape (nx, m)
        Returns: output of the network, and the cache
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            Wl = self.__weights["W{}".format(layer)]
            bl = self.__weights["b{}".format(layer)]
            Al_prev = self.__cache["A{}".format(layer - 1)]

            Zl = np.matmul(Wl, Al_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))  # sigmoid
            self.__cache["A{}".format(layer)] = Al

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Y: numpy.ndarray of shape (1, m) with correct labels
        A: numpy.ndarray of shape (1, m) with activated output
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the deep neural network’s predictions.

        X: numpy.ndarray of shape (nx, m)
        Y: numpy.ndarray of shape (1, m)
        Returns: prediction, cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
