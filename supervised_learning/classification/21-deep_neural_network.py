#!/usr/bin/env python3
"""Deep neural network for binary classification (with gradient descent)"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        nx: number of input features
        layers: list of nodes in each layer of the network
        """
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
        """Number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Dictionary holding intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Dictionary holding all weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation."""
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            Wl = self.__weights["W{}".format(layer)]
            bl = self.__weights["b{}".format(layer)]
            Al_prev = self.__cache["A{}".format(layer - 1)]

            Zl = np.matmul(Wl, Al_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))
            self.__cache["A{}".format(layer)] = Al

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Logistic regression cost."""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the deep neural network.

        Y: numpy.ndarray of shape (1, m)
        cache: dict with all intermediary activations (A0..AL)
        alpha: learning rate
        Updates __weights
        """
        m = Y.shape[1]
        L = self.__L

        # Initialize dZ for the output layer
        dZ = cache["A{}".format(L)] - Y

        # One loop backward over layers L..1
        for layer in range(L, 0, -1):
            Al_prev = cache["A{}".format(layer - 1)]
            Wl = self.__weights["W{}".format(layer)]

            dW = (1 / m) * np.matmul(dZ, Al_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # Update weights and biases
            self.__weights["W{}".format(layer)] = Wl - alpha * dW
            self.__weights["b{}".format(layer)] = (
                self.__weights["b{}".format(layer)] - alpha * db
            )

            if layer > 1:
                # Compute dZ for the previous layer using sigmoid derivative
                Al_prev = cache["A{}".format(layer - 1)]
                dZ = np.matmul(Wl.T, dZ) * (Al_prev * (1 - Al_prev))
