#!/usr/bin/env python3
"""
Fill missing data points according to rules
"""

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Drop Weighted_Price
df = df.drop(columns=['Weighted_Price'])

# Fill Close with previous value
df['Close'] = df['Close'].fillna(method='ffill')

# Fill Open, High, Low with same row Close when missing
for col in ['Open', 'High', 'Low']:
    df[col] = df[col].fillna(df['Close'])

# Fill volumes with 0
for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[col] = df[col].fillna(0)

print(df.head())
print(df.tail())
