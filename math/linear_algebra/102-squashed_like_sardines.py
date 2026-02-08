#!/usr/bin/env python3
# Module: concatenate nested lists along a specified axis (cat_matrices)


def _same_shape(a, b):
    """Return True if nested lists a and b share the same shape.

    Scalars (int/float) are considered matching. Lists are compared
    recursively: lengths and corresponding element shapes must match.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return True
    if not isinstance(a, list) or not isinstance(b, list):
        return False
    if len(a) != len(b):
        return False
    return all(_same_shape(x, y) for x, y in zip(a, b))


def _deep_copy(obj):
    """Return a deep copy of nested lists (no external imports).

    Non-list objects are returned as-is (assumed immutable scalars).
    """
    if isinstance(obj, list):
        return [_deep_copy(x) for x in obj]
    return obj


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate nested lists along the given axis.

    Returns a new nested list (deep copy) on success, or None if the
    matrices' shapes are incompatible for concatenation along axis.
    """
    if axis == 0:
        # both inputs must be lists and their inner shapes must match
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return None
        if len(mat1) == 0 or len(mat2) == 0:
            # concatenation with empty lists is allowed; return copy
            return _deep_copy(mat1) + _deep_copy(mat2)
        if not _same_shape(mat1[0], mat2[0]):
            return None
        return _deep_copy(mat1) + _deep_copy(mat2)

    # deeper axis: lengths of current dimension must match
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if len(mat1) != len(mat2):
        return None

    out = []
    for a, b in zip(mat1, mat2):
        c = cat_matrices(a, b, axis=axis - 1)
        if c is None:
            return None
        out.append(c)
    return out


