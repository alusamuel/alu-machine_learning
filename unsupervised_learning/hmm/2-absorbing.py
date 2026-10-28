#!/usr/bin/env python3
"""Identify absorbing Markov chains."""
import numpy as np


def absorbing(P):
    """Check whether every state can reach an absorbing state."""
    if (not isinstance(P, np.ndarray) or P.ndim != 2
            or P.shape[0] == 0 or P.shape[0] != P.shape[1]
            or not np.all(np.isfinite(P)) or np.any(P < 0)
            or not np.allclose(P.sum(axis=1), 1)):
        return False
    reachable = np.diag(P) == 1
    if not np.any(reachable):
        return False
    for _ in range(P.shape[0]):
        expanded = reachable | np.any((P > 0) & reachable[None, :], axis=1)
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    return bool(np.all(reachable))
