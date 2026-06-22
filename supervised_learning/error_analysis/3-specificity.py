#!/usr/bin/env python3
"""
Module for computing specificity per class from a confusion matrix.
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Parameters
    ----------
    confusion : numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows are
        true labels and columns are predicted labels.

    Returns
    -------
    numpy.ndarray
        Array of shape (classes,) containing specificity for each class.
    """
    if not isinstance(confusion, np.ndarray):
        raise TypeError("confusion must be a numpy.ndarray")

    classes = confusion.shape[0]
    total = np.sum(confusion)

    # True positives per class
    tp = np.diag(confusion)
    # Actual positives per class (rows)
    actual_positives = np.sum(confusion, axis=1)
    # Actual negatives per class
    actual_negatives = total - actual_positives

    # False positives per class (columns sum - TP)
    fp = np.sum(confusion, axis=0) - tp

    # True negatives = all negatives - false positives
    tn = actual_negatives - fp

    with np.errstate(divide='ignore', invalid='ignore'):
        spec = tn / actual_negatives
        spec = np.where(actual_negatives == 0, 0, spec)

    return spec
