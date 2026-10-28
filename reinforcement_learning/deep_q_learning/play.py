#!/usr/bin/env python3
"""Display and score games played by the saved Breakout policy."""
import argparse
from pathlib import Path

import numpy as np
from keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from keras.optimizers import Adam

from train import (BreakoutProcessor, POLICY_PATH, WINDOW_LENGTH,
                   make_environment)


def main():
    """Load policy.h5 and evaluate a purely greedy DQN with live rendering."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=POLICY_PATH)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    if not args.policy.is_file():
        parser.error('policy file not found; run train.py first')
    if args.episodes <= 0:
        parser.error('episodes must be positive')
    env = make_environment(args.seed)
    try:
        model = load_model(str(args.policy), compile=False)
        policy = GreedyQPolicy()
        agent = DQNAgent(
            model=model, nb_actions=env.action_space.n,
            memory=SequentialMemory(limit=WINDOW_LENGTH + 1,
                                    window_length=WINDOW_LENGTH),
            policy=policy, test_policy=policy,
            processor=BreakoutProcessor(clip_rewards=False))
        agent.compile(Adam(lr=0.00025), metrics=['mae'])
        history = agent.test(env, nb_episodes=args.episodes,
                             visualize=not args.no_render, verbose=2)
        scores = np.asarray(history.history['episode_reward'])
        print('Scores:', scores.tolist())
        print('Mean score: {:.2f}; best score: {:.2f}'.format(
            scores.mean(), scores.max()))
        print('Score > 10 achieved: {}'.format(bool(np.any(scores > 10))))
    finally:
        env.close()


if __name__ == '__main__':
    main()
