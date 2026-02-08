#!/usr/bin/env python3
import numpy as np


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes."""
    slices = [slice(None)] * matrix.ndim
    for ax, sl in axes.items():
        slices[ax] = slice(*sl)
    return matrix[tuple(slices)]
