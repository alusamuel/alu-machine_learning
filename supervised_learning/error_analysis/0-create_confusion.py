#!/usr/bin/env python3
"""
Module for creating a confusion matrix from one-hot labels and predictions.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters
    ----------
    labels : numpy.ndarray
        One-hot array of shape (m, classes) with correct labels.
    logits : numpy.ndarray
        One-hot array of shape (m, classes) with predicted labels.

    Returns
    -------
    numpy.ndarray
        Confusion matrix of shape (classes, classes) where rows are
        true labels and columns are predicted labels.
    """
    if (not isinstance(labels, np.ndarray) or
            not isinstance(logits, np.ndarray)):
        raise TypeError("labels and logits must be numpy.ndarray")

    if labels.shape != logits.shape:
        raise ValueError("labels and logits must have the same shape")

    # Get class indices from one-hot encodings
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    # Count occurrences
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
