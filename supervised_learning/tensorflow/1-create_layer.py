#!/usr/bin/env python3
"""Defines a function that creates a single dense layer."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Create a layer.

    Args:
        prev: tensor output of previous layer.
        n: number of nodes in the layer.
        activation: activation function.

    Returns:
        Tensor output of the layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name='layer'
    )
    return layer(prev)
