#!/usr/bin/env python3
"""Backward likelihoods for a hidden Markov model."""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Return the observation likelihood and backward probability table."""
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1
            or Observation.size == 0
            or not np.issubdtype(Observation.dtype, np.integer)
            or not isinstance(Emission, np.ndarray) or Emission.ndim != 2
            or 0 in Emission.shape):
        return None, None
    n, m = Emission.shape
    if (not isinstance(Transition, np.ndarray)
            or Transition.shape != (n, n)
            or not isinstance(Initial, np.ndarray) or Initial.shape != (n, 1)
            or np.any(Observation < 0) or np.any(Observation >= m)):
        return None, None
    for probabilities in (Emission, Transition, Initial.T):
        if (not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1)):
            return None, None
    B = np.ones((n, Observation.size))
    for t in range(Observation.size - 2, -1, -1):
        B[:, t] = Transition @ (Emission[:, Observation[t + 1]] * B[:, t + 1])
    probability = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return probability, B
