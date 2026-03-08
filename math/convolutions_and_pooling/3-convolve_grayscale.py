#!/usr/bin/env python3
"""Convolution on images with channels."""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels.

    images: np.ndarray of shape (m, h, w, c)
    kernel: np.ndarray of shape (kh, kw, c)
    padding: 'same', 'valid', or (ph, pw)
    stride: tuple (sh, sw)

    Returns: np.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        out_h = (h + sh - 1) // sh
        out_w = (w + sw - 1) // sw

        ph_total = max((out_h - 1) * sh + kh - h, 0)
        pw_total = max((out_w - 1) * sw + kw - w, 0)

        ph = ph_total // 2
        pw = pw_total // 2
    else:
        ph, pw = padding

    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    h_p = images_padded.shape[1]
    w_p = images_padded.shape[2]

    out_h = (h_p - kh) // sh + 1
    out_w = (w_p - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            window = images_padded[:, i_start:i_start + kh,
                                   j_start:j_start + kw, :]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output
