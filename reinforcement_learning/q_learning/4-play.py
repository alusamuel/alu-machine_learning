#!/usr/bin/env python3
"""Play a FrozenLake episode using a trained Q-table."""
import numpy as np


def play(env, Q, max_steps=100):
    """Render each visited board and return the total episode reward."""
    initial = env.reset()
    state = initial[0] if isinstance(initial, tuple) else initial
    env.render()
    total_reward = 0
    for _ in range(max_steps):
        action = np.argmax(Q[state])
        result = env.step(action)
        if len(result) == 5:
            state, reward, terminal, truncated, _ = result
            done = terminal or truncated
        else:
            state, reward, done, _ = result
        total_reward += reward
        env.render()
        if done:
            break
    return total_reward
