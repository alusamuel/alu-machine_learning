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
  
