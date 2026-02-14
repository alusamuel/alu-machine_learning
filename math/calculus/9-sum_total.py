#!/usr/bin/env python3
"""
Module for computing the summation of i squared.
"""


def summation_i_squared(n):
    """
    Calculates sum_{i=1}^n i^2.

    Args:
        n (int): stopping condition (must be a positive integer)

    Returns:
        int: value of the summation
        None: if n is not a valid number
    """
    if not isinstance(n, int) or n < 1:
        return None

    # Closed-form formula, no loops used
    return n * (n + 1) * (2 * n + 1) // 6
