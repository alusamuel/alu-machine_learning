#!/usr/bin/env python3
"""Same convolution on grayscale images."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    images: np.ndarray of shape (m, h, w)
    kernel: np.ndarray of shape (kh, kw)

    Returns: np.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = (kh - 1) // 2
    pw = (kw - 1) // 2

    padded = np.pad(
        images,
        ((0, 0), (ph, ph + (kh % 2 == 0)), (pw, pw + (kw % 2 == 0))),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            window = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
