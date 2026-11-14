#!/usr/bin/env python3
"""Softmax policy and Monte Carlo policy-gradient calculations."""
import numpy as np


def policy(matrix, weight):
    """Return action probabilities for each observation row in matrix.

    Subtract the largest logit in each row for numerical stability.
    """
    logits = np.matmul(matrix, weight)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """Sample an action and return its log-policy gradient with respect to W.

    Args:
        state: Single observation with shape (1, number of features).
        weight: Policy weights with shape (number of features, actions).

    Returns:
        The sampled action and a gradient with the same shape as weight.
    """
    probabilities = policy(state, weight)
    action = np.random.choice(probabilities.shape[1], p=probabilities[0])
    log_gradient = -probabilities.copy()
    log_gradient[0, action] += 1
    gradient = np.matmul(state.T, log_gradient)
    return action, gradient
