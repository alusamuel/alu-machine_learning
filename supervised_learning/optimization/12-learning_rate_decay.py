#!/usr/bin/env python3
"""
Inverse time learning rate decay operation in tensorflow.
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay.

    Parameters:
    - alpha: original learning rate
    - decay_rate: decay rate
    - global_step: tf.Variable, steps elapsed
    - decay_step: number of steps before each decay

    Returns:
    - alpha_t: tf.Tensor, decayed learning rate
    """
    alpha_t = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return alpha_t