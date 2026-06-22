#!/usr/bin/env python3
"""
Module for computing F1 score per class from a confusion matrix.
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Parameters
    ----------
    confusion : numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows are
        true labels and columns are predicted labels.

    Returns
    -------
    numpy.ndarray
        Array of shape (classes,) containing the F1 score for each class.
    """
    if not isinstance(confusion, np.ndarray):
        raise TypeError("confusion must be a numpy.ndarray")

    sens = sensitivity(confusion)
    prec = precision(confusion)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * sens * prec / (sens + prec)
        f1 = np.where((sens + prec) == 0, 0, f1)

    return f1
