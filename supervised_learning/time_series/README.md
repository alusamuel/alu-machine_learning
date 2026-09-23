# Time Series Forecasting

This project forecasts the price of Bitcoin (BTC) with a recurrent neural
network, using the Coinbase and Bitstamp datasets.

## Files

- `preprocess_data.py`: cleans the raw data, resamples it to hourly steps,
  normalizes the features and saves the result for training.
- `forecast_btc.py`: builds a `tf.data.Dataset` of 24-hour windows, then
  trains and validates a Keras RNN that predicts the closing price for the
  next hour. It uses mean squared error as the loss.
