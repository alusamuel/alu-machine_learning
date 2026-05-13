#!/usr/bin/env python3
"""Neural network with one hidden layer - forward propagation"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        nx: number of input features
        nodes: number of nodes in the hidden layer
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private hidden layer parameters
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Private output neuron parameters
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for hidden layer weights"""
        return self.__W1

    @property
    def b1(self):
        """Getter for hidden layer bias"""
        return self.__b1

    @property
    def A1(self):
        """Getter for hidden layer activated output"""
        return self.__A1

    @property
    def W2(self):
        """Getter for output neuron weights"""
        return self.__W2

    @property
    def b2(self):
        """Getter for output neuron bias"""
        return self.__b2

    @property
    def A2(self):
        """Getter for output neuron activated output"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network.

        X: numpy.ndarray of shape (nx, m) with input data
        Returns: A1, A2
        """
        # Hidden layer: Z1 = W1.X + b1, A1 = sigmoid(Z1)
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Output neuron: Z2 = W2.A1 + b2, A2 = sigmoid(Z2)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
