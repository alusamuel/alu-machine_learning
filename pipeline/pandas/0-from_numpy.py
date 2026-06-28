#!/usr/bin/env python3
"""
Create a pandas DataFrame from a numpy ndarray
"""

import numpy as np
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    Columns are labeled A, B, C, ... in order.
    """
    n_cols = array.shape[1]
    columns = [chr(ord('A') + i) for i in range(n_cols)]
    df = pd.DataFrame(array, columns=columns)
    return df
