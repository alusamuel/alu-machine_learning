#!/usr/bin/env python3
"""Binomial distribution module.

This module contains a class that represents a Binomial distribution.
"""


class Binomial:
    """Represents a Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a Binomial distribution.

        Args:
            data (list): list of data to estimate the distribution
            n (int): number of Bernoulli trials
            p (float): probability of a success

        Raises:
            ValueError: if n is not a positive value
            ValueError: if p is not in the range (0, 1)
            TypeError: if data is not a list when provided
            ValueError: if data contains fewer than two values
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = float(sum(data)) / len(data)

            # Estimate p first, then n (as they hinted)
            # variance = mean * (1 - p)  => p = 1 - (variance / mean)
            variance = 0
            for x in data:
                variance += (x - mean) ** 2
            variance /= len(data)

            p_est = 1 - (variance / mean)
            n_est = round(mean / p_est)

            p_final = mean / n_est

            self.n = int(n_est)
            self.p = float(p_final)
