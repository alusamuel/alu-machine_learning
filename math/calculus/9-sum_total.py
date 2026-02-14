#!/usr/bin/env python3
"""
Module for computing the summation of i squared.
"""


def summation_i_squared(n):
    """
    Calculates sum_{i=1}^n i^2.

    n: stopping condition
    Returns:
        int: value of the summation
        None: if n is not a valid number
    """
    if not isinstance(n, int) or n < 1:
        return None

    # Formula for 1^2 + 2^2 + ... + n^2
    return n * (n + 1) * (2 * n + 1) // 6
