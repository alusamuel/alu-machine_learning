#!/usr/bin/env python3
"""
Create a batch-normalized dense layer in tensorflow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Parameters:
    - prev: activated output of the previous layer
    - n: number of nodes in the layer
    - activation: activation function

    Returns:
    - A: activated output tensor of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    Z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-8

    mean, var = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(Z, mean, var, beta, gamma, epsilon)

    if activation is None:
        return Z_norm
    return activation(Z_norm)
