#!/usr/bin/env python3
"""Add two nested lists (matrices) element-wise.

This module exposes `add_matrices(mat1, mat2)` which returns a new nested
list containing the element-wise sum of `mat1` and `mat2`. If the inputs
are of different types or shapes, the function returns ``None``.
"""


def add_matrices(mat1, mat2):
    """Adds two matrices of the same shape and returns a new matrix.

    Scalars (int/float) are added directly. For nested lists, addition is
    performed recursively. If the two inputs are not the same type, or their
    shapes differ at any level, ``None`` is returned.
    """
    if type(mat1) is not type(mat2):
        return None
    if isinstance(mat1, (int, float)):
        return mat1 + mat2
    if len(mat1) != len(mat2):
        return None
    result = []
    for a, b in zip(mat1, mat2):
        s = add_matrices(a, b)
        if s is None:
            return None
        result.append(s)
    return result
