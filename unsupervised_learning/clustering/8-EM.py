#!/usr/bin/env python3
"""Expectation maximization for Gaussian mixture models."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Fit a GMM, stopping when the log likelihood changes by at most tol."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or 0 in X.shape):
        return None, None, None, None, None
    if (type(k) is not int or not 0 < k <= X.shape[0]
            or type(iterations) is not int or iterations <= 0
            or type(tol) is not float or not np.isfinite(tol) or tol < 0
            or type(verbose) is not bool):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    previous = None
    for i in range(iterations + 1):
        g, likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        done = (i == iterations or (previous is not None
                and abs(likelihood - previous) <= tol))
        if verbose and (i % 10 == 0 or done):
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(likelihood, 5)))
        if done:
            return pi, m, S, g, likelihood
        previous = likelihood
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None
