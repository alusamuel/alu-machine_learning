#!/usr/bin/env python3
"""Train a softmax policy with Monte Carlo REINFORCE updates."""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Return undiscounted episode scores after training a discrete policy.

    Args:
        env: Gym-compatible environment with vector observations.
        nb_episodes: Number of complete episodes to train on.
        alpha: Learning rate for gradient ascent.
        gamma: Discount factor used to compute reward-to-go.
        show_result: Render each step of every 1000th episode.

    Gradients are collected before updating the weights, then weighted by
    the discounted returns from their corresponding time steps. Both the
    original Gym and the newer Gym/Gymnasium reset and step APIs work.
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []
    for episode in range(nb_episodes):
        initial = env.reset()
        state = initial[0] if isinstance(initial, tuple) else initial
        state = np.asarray(state).reshape(1, -1)
        gradients = []
        rewards = []
        done = False
        render = show_result and (episode + 1) % 1000 == 0

        while not done:
            if render:
                env.render()
            action, gradient = policy_gradient(state, weight)
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
            gradients.append(gradient)
            rewards.append(reward)
            state = np.asarray(next_state).reshape(1, -1)

        discounted_return = 0.0
        for step in range(len(rewards) - 1, -1, -1):
            discounted_return = rewards[step] + gamma * discounted_return
            weight += alpha * discounted_return * gradients[step]

        score = sum(rewards)
        scores.append(score)
        print("Episode: {} Score: {}".format(episode + 1, score),
              end="\r", flush=False)

    return scores
