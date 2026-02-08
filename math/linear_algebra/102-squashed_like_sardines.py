#!/usr/bin/env python3
def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis."""
    if axis == 0:
        if not _same_shape(mat1[0:], mat2[0:]):
            return None
        return [*mat1, *mat2]

    if len(mat1) != len(mat2):
        return None

    out = []
    for a, b in zip(mat1, mat2):
        c = cat_matrices(a, b, axis=axis - 1)
        if c is None:
            return None
        out.append(c)
    return out


def _same_shape(a, b):
    """Checks that two nested lists share the same shape."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return True
    if not isinstance(a, list) or not isinstance(b, list):
        return False
    if len(a) != len(b):
        return False
    return all(_same_shape(x, y) for x, y in zip(a, b))
