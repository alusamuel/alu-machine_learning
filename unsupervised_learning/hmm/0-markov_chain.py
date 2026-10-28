#!/usr/bin/env python3
"""Finite-state Markov chain evolution."""
import numpy as np


def markov_chain(P, s, t=1):
    """Return the state distribution after t transitions."""
    if (not isinstance(P, np.ndarray) or P.ndim != 2
            or P.shape[0] == 0 or P.shape[0] != P.shape[1]
            or not np.all(np.isfinite(P)) or np.any(P < 0)
            or not np.allclose(P.sum(axis=1), 1)):
        return None
    if (not isinstance(s, np.ndarray) or s.shape != (1, P.shape[0])
            or not np.all(np.isfinite(s)) or np.any(s < 0)
            or not np.isclose(s.sum(), 1) or type(t) is not int or t < 0):
        return None
    return s @ np.linalg.matrix_power(P, t)
