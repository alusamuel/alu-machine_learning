# Optimization

This directory contains optimization exercises for supervised learning.
The tasks cover data normalization, data shuffling, mini-batch training,
moving averages, momentum, RMSProp, Adam, learning rate decay, batch
normalization, and building a TensorFlow model that combines these
techniques.

## Files

- `0-norm_constants.py`: calculates feature-wise normalization constants.
- `1-normalize.py`: normalizes data using provided mean and standard deviation.
- `2-shuffle_data.py`: shuffles feature and label matrices together.
- `3-mini_batch.py`: trains a TensorFlow model with mini-batch gradient descent.
- `4-moving_average.py`: calculates an exponentially weighted moving average.
- `5-momentum.py`: updates variables with gradient descent using momentum.
- `6-momentum.py`: creates a TensorFlow momentum optimizer operation.
- `7-RMSProp.py`: updates variables with RMSProp.
- `8-RMSProp.py`: creates a TensorFlow RMSProp optimizer operation.
- `9-Adam.py`: updates variables with Adam optimization.
- `10-Adam.py`: creates a TensorFlow Adam optimizer operation.
- `11-learning_rate_decay.py`: applies inverse time decay to a learning rate.
- `12-learning_rate_decay.py`: creates a TensorFlow decayed learning rate.
- `13-batch_norm.py`: normalizes an activated layer.
- `14-batch_norm.py`: creates a TensorFlow layer with batch normalization.
- `15-model.py`: builds, trains, and saves a TensorFlow model using Adam,
  mini-batches, learning rate decay, and batch normalization.

The `*-main.py` files provide sample usage and basic checks for each task.
