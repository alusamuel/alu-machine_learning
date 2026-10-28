#!/usr/bin/env python3
"""Train a keras-rl DQN policy to play Atari Breakout."""
import argparse
from pathlib import Path

import gym
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, LinearAnnealedPolicy


POLICY_PATH = Path(__file__).resolve().with_name('policy.h5')
WINDOW_LENGTH = 4


class BreakoutProcessor(Processor):
    """Store compact grayscale observations and normalize network inputs."""

    def __init__(self, clip_rewards=True):
        """Preserve original game scores when evaluating the agent."""
        self.clip_rewards = clip_rewards

    def process_observation(self, observation):
        """Convert a screen to an 84-by-84 grayscale uint8 image."""
        image = Image.fromarray(observation).convert('L')
        return np.asarray(image.resize((84, 84)), dtype=np.uint8)

    def process_state_batch(self, batch):
        """Normalize stacked frames only when they enter the network."""
        return batch.astype(np.float32) / 255.0

    def process_reward(self, reward):
        """Clip rewards during training to stabilize temporal errors."""
        if self.clip_rewards:
            return float(np.clip(reward, -1, 1))
        return float(reward)


class FireOnLife(gym.Wrapper):
    """Launch the ball on reset and after each lost life."""

    def __init__(self, env):
        """Find FIRE in the Atari environment's action set."""
        super().__init__(env)
        self.fire = env.unwrapped.get_action_meanings().index('FIRE')
        self.lives = None

    def reset(self, **kwargs):
        """Start a fresh game and launch the first ball."""
        self.env.reset(**kwargs)
        observation, _, done, info = self.env.step(self.fire)
        if done:
            observation = self.env.reset(**kwargs)
        self.lives = info.get('ale.lives')
        return observation

    def step(self, action):
        """Execute an action and relaunch if the game has lives remaining."""
        observation, reward, done, info = self.env.step(action)
        lives = info.get('ale.lives', self.lives)
        if (not done and self.lives is not None and lives is not None
                and lives < self.lives):
            observation, extra, done, info = self.env.step(self.fire)
            reward += extra
            lives = info.get('ale.lives', lives)
        self.lives = lives
        return observation, reward, done, info


def make_environment(seed):
    """Create the legacy four-frame-repeat Breakout environment."""
    env = FireOnLife(gym.make('BreakoutDeterministic-v4'))
    env.seed(seed)
    np.random.seed(seed)
    return env


def build_model(actions):
    """Map four consecutive screens to one Q-value per available action."""
    model = Sequential(name='breakout_policy')
    model.add(Permute((2, 3, 1), input_shape=(WINDOW_LENGTH, 84, 84)))
    for filters, kernel, stride in ((32, 8, 4), (64, 4, 2), (64, 3, 1)):
        model.add(Conv2D(filters, kernel, strides=stride, activation='relu',
                         data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions, training=True, memory_limit=250000):
    """Compile a DQN with replay, a target network, and the selected policy."""
    if training:
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0.1,
            value_test=0.0, nb_steps=1000000)
    else:
        policy = GreedyQPolicy()
    agent = DQNAgent(
        model=model, nb_actions=actions,
        memory=SequentialMemory(limit=memory_limit,
                                window_length=WINDOW_LENGTH),
        policy=policy, test_policy=GreedyQPolicy(),
        processor=BreakoutProcessor(), nb_steps_warmup=50000,
        target_model_update=10000, train_interval=4, gamma=0.99,
        batch_size=32, delta_clip=1.0, enable_double_dqn=True)
    agent.compile(Adam(lr=0.00025), metrics=['mae'])
    return agent


def main():
    """Train, checkpoint, and save the complete final Keras policy model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10000000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--memory-limit', type=int, default=250000)
    parser.add_argument('--resume', type=Path, help='Keras policy to continue')
    args = parser.parse_args()
    if args.steps <= 0 or args.memory_limit <= 50000:
        parser.error('steps must be positive and memory-limit above 50000')
    env = make_environment(args.seed)
    try:
        model = build_model(env.action_space.n)
        if args.resume:
            model.load_weights(str(args.resume))
        agent = build_agent(model, env.action_space.n,
                            memory_limit=args.memory_limit)
        callbacks = [
            ModelIntervalCheckpoint(
                str(POLICY_PATH.with_name('checkpoint_{step}.h5f')),
                interval=250000),
            FileLogger(str(POLICY_PATH.with_name('training.json')),
                       interval=10000)]
        try:
            agent.fit(env, nb_steps=args.steps, visualize=False, verbose=2,
                      callbacks=callbacks, log_interval=10000)
        finally:
            model.save(str(POLICY_PATH), include_optimizer=False)
    finally:
        env.close()


if __name__ == '__main__':
    main()
