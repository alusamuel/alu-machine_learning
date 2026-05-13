#!/usr/bin/env python3
"""Deep neural network for multiclass classification with save/load"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification"""

    # __init__, properties, gradient_descent, train, save, load
    # remain the same as in 26-deep_neural_network.py

    def __init__(self, nx, layers):
        # ... same as 26-deep_neural_network.py ...
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0 or not all(
            isinstance(n, int) and n > 0 for n in layers
        ):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

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
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation of the deep neural network.

        For the last layer, uses softmax for multiclass classification.
        """
        self.__cache["A0"] = X

        # Hidden layers: sigmoid
        for layer in range(1, self.__L):
            Wl = self.__weights["W{}".format(layer)]
            bl = self.__weights["b{}".format(layer)]
            Al_prev = self.__cache["A{}".format(layer - 1)]

            Zl = np.matmul(Wl, Al_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))
            self.__cache["A{}".format(layer)] = Al

        # Output layer: softmax
        WL = self.__weights["W{}".format(self.__L)]
        bL = self.__weights["b{}".format(self.__L)]
        AL_prev = self.__cache["A{}".format(self.__L - 1)]

        ZL = np.matmul(WL, AL_prev) + bL
        # Numerical stability for softmax
        Z_shift = ZL - np.max(ZL, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shift)
        AL = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        self.__cache["A{}".format(self.__L)] = AL

        return AL, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using multiclass cross-entropy.

        Y: one-hot numpy.ndarray of shape (classes, m)
        A: predicted probabilities of shape (classes, m)
        """
        m = Y.shape[1]
        # Add small epsilon for numerical stability
        cost = - (1 / m) * np.sum(Y * np.log(A + 1.0e-8))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the deep neural network’s predictions for multiclass.

        X: numpy.ndarray of shape (nx, m)
        Y: one-hot numpy.ndarray of shape (classes, m)

        Returns:
            prediction: one-hot numpy.ndarray of shape (classes, m)
            cost: multiclass cross-entropy cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        # Predicted class is argmax over classes; return as one-hot matrix
        classes = A.shape[0]
        m = A.shape[1]
        pred_idx = np.argmax(A, axis=0)
        prediction = np.zeros((classes, m))
        prediction[pred_idx, np.arange(m)] = 1

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the deep neural network.

        For the last layer with softmax + cross-entropy, dZ_L = A_L - Y,
        which matches the previous binary case when classes == 1.
        """
        m = Y.shape[1]
        L = self.__L

        dZ = cache["A{}".format(L)] - Y

        for layer in range(L, 0, -1):
            Al_prev = cache["A{}".format(layer - 1)]
            Wl = self.__weights["W{}".format(layer)]

            dW = (1 / m) * np.matmul(dZ, Al_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights["W{}".format(layer)] = Wl - alpha * dW
            self.__weights["b{}".format(layer)] = (
                self.__weights["b{}".format(layer)] - alpha * db
            )

            if layer > 1:
                Al_prev = cache["A{}".format(layer - 1)]
                dZ = np.matmul(Wl.T, dZ) * (Al_prev * (1 - Al_prev))

    def train(self, X, Y,
              iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        # ... same as 26-deep_neural_network.py ...
        import matplotlib.pyplot as plt

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        A, cache = self.forward_prop(X)
        c0 = self.cost(Y, A)
        if verbose:
            print("Cost after {} iterations: {}".format(0, c0))
        if graph:
            costs.append(c0)
            iters.append(0)

        for i in range(1, iterations + 1):
            self.gradient_descent(Y, cache, alpha)
            A, cache = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                c = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
                if graph:
                    costs.append(c)
                    iters.append(i)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        # ... same as 26-deep_neural_network.py ...
        if not isinstance(filename, str):
            return
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        # ... same as 26-deep_neural_network.py ...
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None
