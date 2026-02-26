#!/usr/bin/env python3
"""1-intersection module"""

import numpy as np
likelihood = __import__('0-likelihood').likelihood


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining x and n 
    with each probability in P,
    weighted by the priors in Pr.
    """

    # 1) validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # 2) validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
          "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")

    # 3) validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # 4) validate Pr type/shape
    if not isinstance(Pr, np.ndarray):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # 5) validate ranges for P then Pr (order matters)
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # 6) validate Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # compute intersection = likelihood * prior
    return likelihood(x, n, P) * Pr
