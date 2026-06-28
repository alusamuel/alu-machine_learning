#!/usr/bin/env python3
"""
Transpose and sort DataFrame in reverse chronological order
"""

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.sort_values(by='Timestamp', ascending=False)
df = df.transpose()

print(df.tail(8))
