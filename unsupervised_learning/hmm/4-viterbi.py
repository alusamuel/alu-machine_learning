#!/usr/bin/env python3
"""Most likely hidden-state sequence for a hidden Markov model."""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Return the most likely path and its joint probability."""
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
    length = Observation.size
    V = np.zeros((n, length))
    previous = np.zeros((n, length), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, length):
        candidates = V[:, t - 1, None] * Transition
        previous[:, t] = np.argmax(candidates, axis=0)
        V[:, t] = candidates.max(axis=0) * Emission[:, Observation[t]]
    state = int(np.argmax(V[:, -1]))
    probability = V[state, -1]
    path = [state]
    for t in range(length - 1, 0, -1):
        state = int(previous[state, t])
        path.append(state)
    path.reverse()
    return path, probability
