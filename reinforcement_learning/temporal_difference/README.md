# Temporal difference learning

- `0-monte_carlo.py`: first-visit Monte Carlo policy evaluation.
- `1-td_lambtha.py`: TD(lambda) with accumulating state eligibility traces.
- `2-sarsa_lambtha.py`: SARSA(lambda) with accumulating state-action traces
  and exponentially decaying epsilon-greedy exploration.

All functions update the supplied NumPy table in place and return it. They
support both four-value and five-value environment step results. Terminal
states have no bootstrap value; time-limit truncation retains bootstrapping
for TD and SARSA. Traces reset at the start of each episode.

The supplied FrozenLake examples use a legacy Gym environment. Examples without
an environment seed can produce different results between runs.
