#!/usr/bin/env python3
"""Initialize a tabular action-value function."""
import numpy as np


def q_init(env):
    """Return one zero action value for each environment state-action pair."""
    return np.zeros((env.observation_space.n, env.action_space.n))
