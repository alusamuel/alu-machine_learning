#!/usr/bin/env python3
"""Tabular Q-learning for FrozenLake."""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Train Q in place, penalize holes, and return per-episode rewards."""
    initial_epsilon = epsilon
    total_rewards = []
    for episode in range(episodes):
        initial = env.reset()
        state = initial[0] if isinstance(initial, tuple) else initial
        episode_reward = 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminal, truncated, _ = result
                done = terminal or truncated
            else:
                next_state, reward, done, info = result
                terminal = done and not info.get('TimeLimit.truncated', False)
            if terminal and reward == 0:
                reward = -1
            target = reward
            if not terminal:
                target += gamma * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])
            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)
    return Q, total_rewards
