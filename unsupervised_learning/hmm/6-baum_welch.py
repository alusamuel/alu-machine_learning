#!/usr/bin/env python3
"""Baum-Welch estimation of hidden Markov model parameters."""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Estimate transitions and emissions, keeping initial priors fixed."""
    if (not isinstance(Observations, np.ndarray) or Observations.ndim != 1
            or Observations.size == 0
            or not np.issubdtype(Observations.dtype, np.integer)
            or not isinstance(Emission, np.ndarray) or Emission.ndim != 2
            or 0 in Emission.shape):
        return None, None
    n, m = Emission.shape
    if (not isinstance(Transition, np.ndarray)
            or Transition.shape != (n, n)
            or not isinstance(Initial, np.ndarray) or Initial.shape != (n, 1)
            or np.any(Observations < 0) or np.any(Observations >= m)):
        return None, None
    for probabilities in (Emission, Transition, Initial.T):
        if (not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1)):
            return None, None
    if type(iterations) is not int or iterations <= 0 or Observations.size < 2:
        return None, None
    transition = Transition.astype(float, copy=True)
    emission = Emission.astype(float, copy=True)
    length = Observations.size
    symbols = np.eye(m)[Observations]
    for _ in range(iterations):
        F = np.empty((n, length))
        scales = np.empty(length)
        F[:, 0] = Initial[:, 0] * emission[:, Observations[0]]
        scales[0] = F[:, 0].sum()
        if scales[0] <= 0:
            return None, None
        F[:, 0] /= scales[0]
        for t in range(1, length):
            F[:, t] = (F[:, t - 1] @ transition) * emission[:, Observations[t]]
            scales[t] = F[:, t].sum()
            if scales[t] <= 0:
                return None, None
            F[:, t] /= scales[t]
        B = np.ones((n, length))
        for t in range(length - 2, -1, -1):
            future_emissions = emission[:, Observations[t + 1]] * B[:, t + 1]
            B[:, t] = transition @ future_emissions / scales[t + 1]
        gamma = F * B
        gamma /= gamma.sum(axis=0, keepdims=True)
        future = emission[:, Observations[1:]] * B[:, 1:] / scales[1:]
        counts = transition * (F[:, :-1] @ future.T)
        mass = gamma[:, :-1].sum(axis=1, keepdims=True)
        total = gamma.sum(axis=1, keepdims=True)
        # Keep parameters for states with no posterior occupancy.
        transition = np.divide(counts, mass, out=transition, where=mass > 0)
        emission = np.divide(gamma @ symbols, total, out=emission,
                             where=total > 0)
    return transition, emission
