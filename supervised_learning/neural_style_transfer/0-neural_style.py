#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST with style and content images."""
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescale image so max side is 512 and pixel values are in [0, 1]."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if max(h, w) > 512:
            if h >= w:
                new_h = 512
                new_w = int(w * (512 / h))
            else:
                new_w = 512
                new_h = int(h * (512 / w))
        else:
            new_h, new_w = h, w

        image = tf.image.resize_images(
            image[np.newaxis, ...],
            [new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC,
            align_corners=False
        )
        image = image / 255.0
        return image