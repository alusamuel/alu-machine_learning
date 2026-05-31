#!/usr/bin/env python3
"""
Adam optimization update (NumPy).
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Parameters:
    - alpha: learning rate
    - beta1: first moment weight
    - beta2: second moment weight
    - epsilon: small number to avoid division by zero
    - var: np.ndarray, variable to update
    - grad: np.ndarray, gradient of var
    - v: np.ndarray, previous first moment
    - s: np.ndarray, previous second moment
    - t: int, time step (starting at 1)

    Returns:
    - var_updated, v_new, s_new
    """
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corr = v_new / (1 - beta1 ** t)
    s_corr = s_new / (1 - beta2 ** t)

    var_updated = var - alpha * v_corr / (np.sqrt(s_corr) + epsilon)
    return var_updated, v_new, s_new
