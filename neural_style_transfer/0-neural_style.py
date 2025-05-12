#!/usr/bin/env python3
"""Module for Neural Style Transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer

        Args:
            style_image (ndarray): Image used as style reference
            content_image (ndarray): Image used as content reference
            alpha (float): Weight for content cost
            beta (float): Weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
           style_image.shape[-1] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or \
           content_image.shape[-1] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
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
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels

        Args:
            image (ndarray): Image to be scaled

        Returns:
            tensor: Scaled image
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim
        new_shape = (int(h * scale), int(w * scale))

        image = image.astype('float32')
        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_bicubic(image, new_shape)

        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image
