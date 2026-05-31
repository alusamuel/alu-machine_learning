#!/usr/bin/env python3
"""
Batch normalization for an unactivated layer output (NumPy).
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output using batch normalization.

    Parameters:
    - Z: np.ndarray of shape (m, n)
    - gamma: np.ndarray of shape (1, n)
    - beta: np.ndarray of shape (1, n)
    - epsilon: small number to avoid division by zero

    Returns:
    - Z_tilde: normalized and scaled output
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde
