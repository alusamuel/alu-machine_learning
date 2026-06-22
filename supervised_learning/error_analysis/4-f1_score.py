#!/usr/bin/env python3
"""
Module for computing F1 score per class from a confusion matrix.
"""

import numpy as np


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

    tp = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    predicted_positives = np.sum(confusion, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        sens = tp / actual_positives
        sens = np.where(actual_positives == 0, 0, sens)
        prec = tp / predicted_positives
        prec = np.where(predicted_positives == 0, 0, prec)
        f1 = 2 * sens * prec / (sens + prec)
        f1 = np.where((sens + prec) == 0, 0, f1)

    return f1
