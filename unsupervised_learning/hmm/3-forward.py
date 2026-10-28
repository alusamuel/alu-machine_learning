#!/usr/bin/env python3
"""Forward likelihoods for a hidden Markov model."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Return the observation likelihood and forward probability table."""
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
    F = np.zeros((n, Observation.size))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, Observation.size):
        F[:, t] = (F[:, t - 1] @ Transition) * Emission[:, Observation[t]]
    return F[:, -1].sum(), F
