#!/usr/bin/env python3
"""
Load data from a file into a pandas DataFrame
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame.

    filename: path to file
    delimiter: column separator
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
