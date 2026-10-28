#!/usr/bin/env python3
"""First-visit Monte Carlo policy evaluation."""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Update V from complete sampled returns using a constant step size."""
    for _ in range(episodes):
        initial = env.reset()
        state = initial[0] if isinstance(initial, tuple) else initial
        trajectory = []
        for _ in range(max_steps):
            result = env.step(policy(state))
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
            trajectory.append((state, reward))
            state = next_state
            if done:
                break
        returns = [0.0] * len(trajectory)
        total = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            total = trajectory[t][1] + gamma * total
            returns[t] = total
        visited = set()
        for t, (state, _) in enumerate(trajectory):
            if state not in visited:
                V[state] += alpha * (returns[t] - V[state])
                visited.add(state)
    return V
