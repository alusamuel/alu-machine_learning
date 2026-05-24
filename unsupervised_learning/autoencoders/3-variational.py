#!/usr/bin/env python3
"""Variational autoencoder.
"""

import tensorflow.keras as keras


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    Args:
        args: (z_mean, z_log_sigma)

    Returns:
        Sampled latent vector.
    """
    z_mean, z_log_sigma = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims: int, dimensionality of the input.
        hidden_layers: list of ints, sizes of hidden layers in encoder.
        latent_dims: int, dimensionality of latent space.

    Returns:
        encoder: encoder model (outputs z, z_mean, z_log_sigma).
        decoder: decoder model.
        auto: full VAE model.
    """
    # Encoder
    inputs = keras.layers.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None, name='z_mean')(x)
    z_log_sigma = keras.layers.Dense(
        latent_dims, activation=None, name='z_log_sigma'
    )(x)
    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,), name='z'
    )([z_mean, z_log_sigma])

    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma], name='encoder')

    # Decoder
    latent_inputs = keras.layers.Input(
        shape=(latent_dims,), name='z_sampling'
    )
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # VAE model
    outputs = decoder(z)
    auto = keras.Model(inputs, outputs, name='variational_autoencoder')

    # KL divergence
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_sigma - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_sigma),
        axis=1
    )

    auto.add_loss(keras.backend.mean(kl_loss))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
