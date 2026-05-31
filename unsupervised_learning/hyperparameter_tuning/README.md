# Hyperparameter Tuning

This directory contains exercises for hyperparameter tuning with Gaussian
Processes and Bayesian optimization. The implementations focus on a noiseless
one-dimensional Gaussian Process and use expected improvement to choose new
sample points for optimizing a black-box function.

## Learning Objectives

- Build a Gaussian Process model for one-dimensional data.
- Compute covariance matrices with the radial basis function kernel.
- Predict mean and variance for unseen sample points.
- Update a Gaussian Process with newly observed data.
- Use expected improvement as an acquisition function.
- Run Bayesian optimization for minimization or maximization problems.

## Files

- `0-gp.py`: defines a `GaussianProcess` class and computes its covariance
  kernel.
- `1-gp.py`: adds prediction of the mean and variance for candidate points.
- `2-gp.py`: adds updates for new observations.
- `3-bayes_opt.py`: initializes Bayesian optimization with a black-box
  function, Gaussian Process, bounds, acquisition samples, and optimization
  mode.
- `4-bayes_opt.py`: implements the expected improvement acquisition function.
- `5-bayes_opt.py`: adds the full optimization loop and returns the best
  observed input and output.

## Requirements

The scripts use:

- Python 3
- NumPy
- SciPy, for the normal distribution functions used by expected improvement

## Usage

Each file is designed as a standalone task and can be imported by its matching
test script. Later files build on earlier implementations, so `5-bayes_opt.py`
contains the most complete Bayesian optimization workflow.

Example import:

```python
BayesianOptimization = __import__('5-bayes_opt').BayesianOptimization
```

The optimizer expects an initial set of sample points and observations, bounds
for the search space, and a number of acquisition samples to evaluate.
