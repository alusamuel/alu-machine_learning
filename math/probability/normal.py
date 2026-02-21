#!/usr/bin/env python3
"""Normal distribution module.

This module contains a class that represents a Normal distribution.
"""


class Normal:
    """Represents a Normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a Normal distribution.

        Args:
            data (list): list of data to estimate the distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution

        Raises:
            ValueError: if stddev is not a positive value when dt is None
            TypeError: if data is not a list when provided
            ValueError: if data contains fewer than two values
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            mean_value = sum(data) / n

            variance = 0
            for x in data:
                variance += (x - mean_value) ** 2
            variance /= n

            self.mean = float(mean_value)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score."""
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """Calculates the PDF value for a given x-value.

        Args:
            x (float): x-value

        Returns:
            float: PDF value for x
        """
        pi = 3.1415926536
        e = 2.7182818285

        z = (x - self.mean) / self.stddev
        exponent = e ** (-0.5 * (z ** 2))
        denominator = self.stddev * ((2 * pi) ** 0.5)

        return (1 / denominator) * exponent

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value."""
        e = 2.718281828459045

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        t = 1.0 / (1.0 + 0.5 * abs(z))

        tau = t * (e ** (
            -z * z
            - 1.26551223
            + 1.00002368 * t
            + 0.37409196 * (t ** 2)
            + 0.09678418 * (t ** 3)
            - 0.18628806 * (t ** 4)
            + 0.27886807 * (t ** 5)
            - 1.13520398 * (t ** 6)
            + 1.48851587 * (t ** 7)
            - 0.82215223 * (t ** 8)
            + 0.17087277 * (t ** 9)
        ))

        erf = 1 - tau

        if z < 0:
            erf = -erf

        return 0.5 * (1 + erf)
