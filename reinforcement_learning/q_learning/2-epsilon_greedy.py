#!/usr/bin/env python3
"""Epsilon-greedy action selection."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Explore with probability epsilon; otherwise choose a greedy action."""
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(0, Q.shape[1])
