# Hidden Markov models

NumPy implementations of Markov chain evolution, regular-chain stationary
distributions, absorbing-chain detection, forward and backward likelihoods,
Viterbi decoding, and Baum-Welch parameter estimation.

Baum-Welch uses scaled forward and backward probabilities to reduce underflow.
The starting distribution remains fixed, and input parameters are not modified.
