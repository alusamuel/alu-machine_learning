#!/usr/bin/env python3
"""
Module for computing a bias-corrected exponentially weighted moving average.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.

    Parameters:
    - data: list of floats
    - beta: float, weight for moving average

    Returns:
    - m_avg: list of moving averages
    """
    m_avg = []
    v = 0.0
    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        m_avg.append(v_corrected)
    return m_avg
