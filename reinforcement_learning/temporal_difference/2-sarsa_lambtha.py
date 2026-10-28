#!/usr/bin/env python3
"""On-policy SARSA(lambda) with epsilon-greedy exploration."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Update Q with accumulating traces and exponential epsilon decay."""
    def choose_action(state, threshold):
        """Sample an epsilon-greedy action from the current Q table."""
        if np.random.uniform() < threshold:
            return np.random.randint(Q.shape[1])
        return int(np.argmax(Q[state]))

    initial_epsilon = epsilon
    for episode in range(episodes):
        initial = env.reset()
        state = initial[0] if isinstance(initial, tuple) else initial
        action = choose_action(state, epsilon)
        traces = np.zeros_like(Q, dtype=float)
        for _ in range(max_steps):
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminal, truncated, _ = result
                done = terminal or truncated
            else:
                next_state, reward, done, info = result
                terminal = done and not info.get('TimeLimit.truncated', False)
            next_action = None
            target = reward
            if not terminal:
                next_action = choose_action(next_state, epsilon)
                target += gamma * Q[next_state, next_action]
            delta = target - Q[state, action]
            traces[state, action] += 1
            Q += alpha * delta * traces
            traces *= gamma * lambtha
            state, action = next_state, next_action
            if done:
                break
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * (episode + 1))
    return Q
