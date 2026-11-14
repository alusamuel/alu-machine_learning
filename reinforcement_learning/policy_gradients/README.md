# Policy Gradients

Train a linear softmax policy to balance CartPole using Monte Carlo
REINFORCE and NumPy.

## Tasks

| Task | File | Function |
| --- | --- | --- |
| 0. Simple policy | `policy_gradient.py` | `policy(matrix, weight)` |
| 1. Monte Carlo policy gradient | `policy_gradient.py` | `policy_gradient(state, weight)` |
| 2. Training | `train.py` | `train(env, nb_episodes, alpha=0.000045, gamma=0.98)` |
| 3. Animation | `train.py` | Add `show_result=False` to `train` |

`policy` computes a numerically stable softmax of the observation-weight
product. `policy_gradient` samples an action from those probabilities and
returns the gradient of its log probability:

```text
gradient = state.T @ (one_hot(action) - probabilities)
```

Training initializes random weights, collects a complete episode, and applies
gradient ascent using each step's discounted reward-to-go. The returned list
contains the undiscounted reward sum for every episode. The episode number and
score are printed on the same line as training progresses. With
`show_result=True`, every step of episodes 1000, 2000, and so on is rendered.

## Run

Install NumPy, Matplotlib, and either Gym (for the assignment's examples) or
Gymnasium with its classic-control dependencies. Run the following from this
directory for the Gymnasium API:

```python
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from train import train

np.random.seed(1)
env = gym.make("CartPole-v1")
try:
    scores = train(env, 10000)
finally:
    env.close()

plt.plot(np.arange(len(scores)), scores)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.show()
```

The functions also accept the original Gym API used by the supplied main
files. For animation, pass `show_result=True`; modern Gym/Gymnasium also
requires `render_mode="human"` when constructing the environment. Human render
mode may display every episode automatically, depending on the environment;
the training function itself calls `render()` only every 1000th episode.
Learning curves vary with the random seed, learning rate, and discount factor.

## Validation

Both supplied policy examples match their expected outputs. Numerical
derivatives confirm the log-policy gradients, and deterministic environment
checks cover discounted returns, both Gym API formats, termination,
truncation, and the rendering interval. The Python files pass `pycodestyle`.

A 10,000-episode Gymnasium CartPole-v1 run with the default parameters and
NumPy/environment seed 1 increased the mean score from 21.40 in the first 100
episodes to 455.79 in the final 100, reaching the 500-step episode limit.
Graphical animation was not visually inspected.
