#!/usr/bin/env python3
"""Strided convolution on grayscale images."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images.

    images: np.ndarray of shape (m, h, w)
    kernel: np.ndarray of shape (kh, kw)
    padding: 'same', 'valid', or (ph, pw)
    stride: tuple (sh, sw)

    Returns: np.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        # derive padding height
        if (h - kh) % sh == 0:
            ph = max((kh - sh) // 2, 0)
        else:
            ph = max((kh - sh) // 2 + 1, 0)
        # derive padding width
        if (w - kw) % sw == 0:
            pw = max((kw - sw) // 2, 0)
        else:
            pw = max((kw - sw) // 2 + 1, 0)
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )
    h_p = h + 2 * ph
    w_p = w + 2 * pw

    out_h = (h_p - kh) // sh + 1
    out_w = (w_p - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * sh
            j_start = j * sw
            window = padded[:, i_start:i_start + kh, j_start:j_start + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
