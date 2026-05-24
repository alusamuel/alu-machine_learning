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

    # Latent representation
    # latent_dims is a tuple; we ensure the shape matches by using Conv2D
    latent = keras.layers.Conv2D(
        filters=latent_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(x)
    encoder = keras.Model(inputs, latent, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    # For the decoder, we mirror the encoder structure, using Conv2D + UpSampling2D
    for i, f in enumerate(reversed(filters)):
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        # All but the last two convs use same padding + upsampling
        if i < len(filters) - 2:
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        elif i == len(filters) - 2:
            # second to last conv: valid padding + upsample
            x = keras.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                padding='valid',
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