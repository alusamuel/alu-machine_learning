#!/usr/bin/env python3
"""Load the assignment's FrozenLake environment."""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Create FrozenLake with a custom, named, or random default map."""
    return gym.make('FrozenLake-v0', desc=desc, map_name=map_name,
                    is_slippery=is_slippery)
