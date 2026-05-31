#!/usr/bin/env python3
"""
RMSProp optimization update (NumPy).
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    - alpha: learning rate
    - beta2: RMSProp weight
    - epsilon: small number to avoid division by zero
    - var: np.ndarray, variable to update
    - grad: np.ndarray, gradient of var
    - s: np.ndarray, previous second moment

    Returns:
    - var_updated, s_new
    """
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    var_updated = var - alpha * grad / (np.sqrt(s_new) + epsilon)
    return var_updated, s_new