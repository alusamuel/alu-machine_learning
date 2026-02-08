#!/usr/bin/env python3
# Module: slice a numpy.ndarray along specified axes without importing modules


def np_slice(matrix, axes=None):
    """Return a new numpy.ndarray that is a slice of `matrix`.

    `axes` should be a dict mapping axis index -> slice specification.
    The slice specification may be:
    - an int (select a single index along that axis),
    - a 1-tuple/list like (i,) which selects a single index i,
    - a 2- or 3-tuple/list like (start, stop) or (start, stop, step) which
      will be passed to slice(*spec).

    The function does not import numpy; it slices the provided array and
    returns a copy so the result is a new numpy.ndarray (if `matrix` is one).
    """
    if axes is None:
        axes = {}

    slices = [slice(None)] * matrix.ndim
    for ax, spec in axes.items():
        # single integer index
        if isinstance(spec, int):
            slices[ax] = spec
            continue

        # treat single-element sequences as an index
        if isinstance(spec, (list, tuple)) and len(spec) == 1:
            slices[ax] = spec[0]
            continue

        # sequence with 2 or 3 items -> create a slice
        if isinstance(spec, (list, tuple)):
            slices[ax] = slice(*spec)
            continue

        # fallback: try to create a slice (handles None etc.)
        slices[ax] = slice(spec)

    return matrix[tuple(slices)].copy()
