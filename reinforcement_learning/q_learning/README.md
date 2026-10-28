# Q-learning

Load FrozenLake, initialize a Q-table, select epsilon-greedy actions, train with
Q-learning, and render a greedy evaluation episode.

The loader follows the assignment's legacy OpenAI Gym `FrozenLake-v0` interface.
It requires a Gym release that provides this environment identifier. Training
and playback accept both four-value and five-value step results.

Training assigns a reward of -1 for falling into a hole and decays exploration
exponentially after each episode. Evaluation returns the environment's original
rewards without the training penalty.
