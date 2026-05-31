#!/usr/bin/env python3
"""
Inverse time learning rate decay implemented in NumPy.
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay (stepwise).

    Parameters:
    - alpha: original learning rate
    - decay_rate: decay rate
    - global_step: number of gradient steps elapsed
    - decay_step: number of steps before next decay

    Returns:
    - alpha_t: updated learning rate
    """
    k = global_step // decay_step
    alpha_t = alpha / (1 + decay_rate * k)
    return alpha_t