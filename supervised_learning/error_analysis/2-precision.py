#!/usr/bin/env python3
"""
Module for computing precision per class from a confusion matrix.
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Parameters
    ----------
    confusion : numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows are
        true labels and columns are predicted labels.

    Returns
    -------
    numpy.ndarray
        Array of shape (classes,) containing precision for each class.
    """
    if not isinstance(confusion, np.ndarray):
        raise TypeError("confusion must be a numpy.ndarray")

    # True positives are the diagonal
    tp = np.diag(confusion)
    # For each class, predicted positives = sum of its column
    predicted_positives = np.sum(confusion, axis=0)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = tp / predicted_positives
        prec = np.where(predicted_positives == 0, 0, prec)

    return prec
