#!/usr/bin/env python3
"""Convolutional autoencoder.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    Args:
        input_dims: tuple of ints, (H, W, C) for input.
        filters: list of ints, number of filters per conv layer in encoder.
        latent_dims: tuple of ints, (h, w, c) for latent space.

    Returns:
        encoder: encoder model.
        decoder: decoder model.
        auto: full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder = keras.Model(inputs, x, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    for i, f in enumerate(reversed(filters)):
        padding = 'valid' if i == len(filters) - 1 else 'same'
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding=padding,
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Last conv: same number of channels as input, sigmoid, no upsampling
    outputs = keras.layers.Conv2D(
        filters=input_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # Full autoencoder
    auto_input = inputs
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name='conv_autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
