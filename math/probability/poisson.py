#!/usr/bin/env python3
"""
Poisson validator module.
"""


def validate_poisson_data(data, lam):
    """
    Validate Poisson data.

    Args:
        data (list): List of occurrences.
        lam (float): Expected rate.

    Returns:
        bool: True if valid.
    """
    if lam <= 0:
        return False
    return all(x > 0 for x in data)
