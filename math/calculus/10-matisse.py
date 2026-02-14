#!/usr/bin/env python3
"""
Module for computing the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    poly: list of coefficients where index is power of x
    Returns:
        list: coefficients of the derivative
        None: if poly is not valid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for c in poly:
        if not isinstance(c, (int, float)):
            return None

    # Constant polynomial
    if len(poly) == 1:
        return [0]

    deriv = []
    for power in range(1, len(poly)):
        deriv.append(power * poly[power])

    # If all coefficients are 0, return [0]
    if all(c == 0 for c in deriv):
        return [0]

    return deriv
