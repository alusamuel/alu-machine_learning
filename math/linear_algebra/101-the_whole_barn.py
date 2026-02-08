#!/usr/bin/env python3
def add_matrices(mat1, mat2):
    """Adds two matrices of the same shape."""
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
