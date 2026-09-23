# Recurrent Neural Networks

This directory contains recurrent neural network exercises built from
scratch with NumPy. The tasks cover simple RNN cells, gated units (GRU and
LSTM), stacking cells into deep RNNs, and bidirectional RNNs.

## Files

- `0-rnn_cell.py`: defines `RNNCell`, a simple RNN cell with a tanh hidden
  state and a softmax output.
- `1-rnn.py`: performs forward propagation for a simple RNN over all time
  steps.
- `2-gru_cell.py`: defines `GRUCell`, a gated recurrent unit with update and
  reset gates.
- `3-lstm_cell.py`: defines `LSTMCell`, an LSTM unit with forget, update and
  output gates.
- `4-deep_rnn.py`: performs forward propagation for a deep (stacked) RNN.
- `5-bi_forward.py`: defines `BidirectionalCell` with the forward direction.
- `6-bi_backward.py`: adds the backward direction to `BidirectionalCell`.
- `7-bi_output.py`: adds `output`, which computes the softmax outputs from
  the concatenated hidden states.
- `8-bi_rnn.py`: performs forward propagation for a bidirectional RNN.

Every cell stores its weights as `W*` (initialized from a standard normal
distribution) and its biases as `b*` (initialized to zeros).
