#!/usr/bin/env python3
"""TD(lambda) policy evaluation with accumulating eligibility traces."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Update V online, resetting eligibility traces for every episode."""
    for _ in range(episodes):
        initial = env.reset()
        state = initial[0] if isinstance(initial, tuple) else initial
        traces = np.zeros_like(V, dtype=float)
        for _ in range(max_steps):
            result = env.step(policy(state))
            if len(result) == 5:
                next_state, reward, terminal, truncated, _ = result
                done = terminal or truncated
            else:
                next_state, reward, done, info = result
                terminal = done and not info.get('TimeLimit.truncated', False)
            target = reward + (0 if terminal else gamma * V[next_state])
            delta = target - V[state]
            traces[state] += 1
            V += alpha * delta * traces
            traces *= gamma * lambtha
            state = next_state
            if done:
                break
    return V
