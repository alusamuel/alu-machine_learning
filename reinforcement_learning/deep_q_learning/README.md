# Breakout DQN

`train.py` builds a convolutional Keras policy and trains it with keras-rl's
`DQNAgent`, `SequentialMemory`, and an annealed `EpsGreedyQPolicy`. It uses
84x84 grayscale frames, four-frame histories, Double DQN, a target network,
and clipped training rewards. The environment launches a new ball after reset
and lost lives so a greedy policy cannot stall waiting to serve.

`play.py` loads the complete network from `policy.h5`, uses `GreedyQPolicy`,
and displays games by default. It reports original game scores, their mean,
and whether any evaluation game scored above 10.

## Setup

These scripts target the legacy Gym / standalone Keras APIs required by the
assignment. Use Linux with Python 3.7 and an isolated environment. The pinned
requirements are a suggested legacy stack and have not been installed or
verified in this workspace. Modern Keras 3 and Gymnasium are not substitutes
for these interfaces. The Atari environment also requires the Breakout ROM.

```sh
python3.7 -m venv .venv
. .venv/bin/activate
pip install 'pip<24.1' 'setuptools<66' wheel
pip install -r requirements.txt
python train.py
python play.py
```

Training defaults to 10 million agent steps and a replay capacity of 250,000
frames (roughly 1.8 GB for frame pixels alone). Plan for substantial additional
RAM, GPU memory, and training time. Short runs check execution but do not
establish a useful policy.

```sh
python train.py --steps 10000000 --seed 0
python train.py --resume policy.h5 --steps 5000000
python play.py --episodes 10
python play.py --episodes 10 --no-render
```

The final complete model is saved beside the scripts as `policy.h5`, including
when training is interrupted after it starts. Periodic checkpoints contain
weights only (`checkpoint_<step>.h5f`); `--resume` accepts either format but
starts fresh replay memory, optimizer state, and exploration scheduling.
Rendering requires a graphical display. Generated models and logs are ignored
by Git because they are large artifacts.

## Validation status

No training has been run and no `policy.h5` has been generated in this
workspace: its Python 3.15 environment lacks the legacy runtime and Atari ROM.
A score above 10 must be verified by running training and evaluation; the
scripts do not guarantee or claim that score without an evaluated model.

Implementation reference: the official keras-rl Atari example and DQN API:
https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py
