#!/usr/bin/env python3
"""Defines a function that creates the forward propagation graph."""

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Create the forward propagation graph.

    Args:
        x: input placeholder.
        layer_sizes: list of layer sizes.
        activations: list of activation functions.

    Returns:
        Prediction tensor.
    """
    output = x
    for size, activation in zip(layer_sizes, activations):
        output = create_layer(output, size, activation)
    return output
