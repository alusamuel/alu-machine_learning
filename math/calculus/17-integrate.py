#!/usr/bin/env python3
"""
Module for computing the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    poly: list of coefficients where index is power of x
    C: integer representing the integration constant

    Returns:
        list: coefficients of an antiderivative of the polynomial
        None: if poly or C are not valid
    """
    # Validate C
    if not isinstance(C, int):
        return None

    # Validate poly
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for c in poly:
        if not isinstance(c, (int, float)):
            return None

    # Start with the constant of integration
    integral = [C]

    # For each coefficient a_k at index k, add a_k / (k + 1) at index k + 1
    for power, coef in enumerate(poly):
        new_coef = coef / (power + 1)

        # If the new coefficient is a whole number, store it as int
        if new_coef.is_integer():
            new_coef = int(new_coef)

        integral.append(new_coef)

    # Remove unnecessary trailing zeros to make list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
