#!/usr/bin/env python3
"""Defines a function that creates placeholders for the input data and labels."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Create placeholders x and y.

    Args:
        nx: number of feature columns.
        classes: number of classes.

    Returns:
        x, y: placeholders for input data and one-hot labels.
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y