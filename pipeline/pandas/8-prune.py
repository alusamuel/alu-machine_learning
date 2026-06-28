#!/usr/bin/env python3
"""
Remove rows where Close is NaN
"""

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df[df['Close'].notna()]

print(df.head())
