#!/usr/bin/env python3
"""
Visualize daily aggregated data from 2017 onwards
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price
df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp to Date and convert to datetime date
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index on Date
df = df.set_index('Date')

# Fill Close with previous value
df['Close'] = df['Close'].fillna(method='ffill')

# Fill Open, High, Low with same row Close
for col in ['Open', 'High', 'Low']:
    df[col] = df[col].fillna(df['Close'])

# Fill volume columns with 0
for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[col] = df[col].fillna(0)

# Keep data from 2017 onwards
df_2017 = df.loc['2017':]

# Resample daily and aggregate
daily = df_2017.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum',
})

# Simple visualization example: plot Close
plt.figure(figsize=(12, 6))
plt.plot(daily.index, daily['Close'])
plt.title('Daily Mean Close Price (2017+)')
plt.xlabel('Date')
plt.ylabel('Close (mean)')
plt.tight_layout()
plt.show()
