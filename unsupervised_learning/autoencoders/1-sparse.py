#!/usr/bin/env python3
"""Sparse autoencoder.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder.

    Args:
        input_dims: int, dimensionality of the input.
        hidden_layers: list of ints, sizes of hidden layers in encoder.
        latent_dims: int, dimensionality of latent space.
        lambtha: float, L1 regularization parameter on the latent output.

    Returns:
        encoder: encoder model.
        decoder: decoder model.
        auto: sparse autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.Model(inputs, latent, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # Full autoencoder
    auto_input = inputs
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name='sparse_autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
