#!/usr/bin/env python3
"""Variational autoencoder.
"""

import tensorflow.keras as keras
import tensorflow as tf


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    Args:
        args: (z_mean, z_log_var)

    Returns:
        Sampled latent vector.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims: int, dimensionality of the input.
        hidden_layers: list of ints, sizes of hidden layers in encoder.
        latent_dims: int, dimensionality of latent space.

    Returns:
        encoder: encoder model (outputs z, z_mean, z_log_var).
        decoder: decoder model.
        auto: full VAE model.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None, name='z_mean')(x)
    z_log_var = keras.layers.Dense(
        latent_dims, activation=None, name='z_log_var'
    )(x)
    z = keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,), name='z_sampling')
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # VAE model
    outputs = decoder(z)
    auto = keras.Model(inputs, outputs, name='variational_autoencoder')

    # Reconstruction loss
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=1)

    # KL divergence
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1
    )

    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto