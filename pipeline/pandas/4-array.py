#!/usr/bin/env python3
"""
Take last 10 rows of High and Close as numpy array
"""

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

A = df[['High', 'Close']].tail(10).to_numpy()

print(A)
