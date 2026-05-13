#!/usr/bin/env python3
"""Neural network with one hidden layer - gradient descent"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        nx: number of input features
        nodes: number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network.

        X: numpy.ndarray of shape (nx, m)
        Returns: A1, A2
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A2):
        """
        Calculates the cost of the model using logistic regression.

        Y: numpy.ndarray of shape (1, m)
        A2: numpy.ndarray of shape (1, m)
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A2) + (1 - Y) * np.log(1.0000001 - A2)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        X: numpy.ndarray of shape (nx, m)
        Y: numpy.ndarray of shape (1, m)
        Returns: (prediction, cost)
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        X: numpy.ndarray of shape (nx, m)
        Y: numpy.ndarray of shape (1, m)
        A1: output of the hidden layer
        A2: predicted output
        alpha: learning rate
        Updates: __W1, __b1, __W2, __b2
        """
        m = Y.shape[1]

        # Output layer gradients
        dZ2 = A2 - Y                           # (1, m)
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)   # (1, nodes)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (1, 1)

        # Hidden layer gradients
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))   # (nodes, m)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)                   # (nodes, nx)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)    # (nodes, 1)

        # Parameter update
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
