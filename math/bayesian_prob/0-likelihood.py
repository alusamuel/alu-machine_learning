#!/usr/bin/env python3
"""0-likelihood module"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining x successes out of n trials
    for each hypothetical probability in P (binomial distribution).
    """

    # validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            """x must be an integer that is greater than or equal to 0"""
        )
    if x > n:
        raise ValueError("x cannot be greater than n")

    # validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # compute binomial coefficient C(n, x) without huge factorials
    k = min(x, n - x)
    comb = 1.0
    for i in range(1, k + 1):
        comb *= (n - k + i) / i  # stable multiplicative form

    # vectorized likelihoods
    return comb * (P ** x) * ((1 - P) ** (n - x))
