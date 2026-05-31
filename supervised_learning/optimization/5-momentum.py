#!/usr/bin/env python3
"""
Gradient descent with momentum update (NumPy).
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum.

    Parameters:
    - alpha: learning rate
    - beta1: momentum weight
    - var: np.ndarray, variable to be updated
    - grad: np.ndarray, gradient of var
    - v: np.ndarray, previous first moment

    Returns:
    - var_updated, v_new
    """
    v_new = beta1 * v + (1 - beta1) * grad
    var_updated = var - alpha * v_new
    return var_updated, v_new