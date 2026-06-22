#!/usr/bin/env python3
"""
Module for computing sensitivity (recall) per class from a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters
    ----------
    confusion : numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows are
        true labels and columns are predicted labels.

    Returns
    -------
    numpy.ndarray
        Array of shape (classes,) containing sensitivity for each class.
    """
    if not isinstance(confusion, np.ndarray):
        raise TypeError("confusion must be a numpy.ndarray")

    # True positives are the diagonal
    tp = np.diag(confusion)
    # For each class, actual positives = sum of its row
    actual_positives = np.sum(confusion, axis=1)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = tp / actual_positives
        sens = np.where(actual_positives == 0, 0, sens)

    return sens
