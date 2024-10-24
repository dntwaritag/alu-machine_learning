#!/usr/bin/env python3

'''
    This script performs a convolution on grayscale images
    with custom padding
'''

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Shape (m, h, w),
        containing multiple grayscale images.
        kernel (numpy.ndarray): Shape (kh, kw),
        containing the kernel for the convolution.
        padding (tuple): (ph, pw),
        containing the padding for the height and width of the image.

    Returns:
        numpy.ndarray: The convolved images.
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Apply padding
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)),
        mode='constant', constant_values=0)

    # Calculate dimensions of the output image
    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1

    # Initialize output array
    convolved_images = np.zeros((m, new_h, new_w))

    # Perform convolution
    for i in range(new_h):
        for j in range(new_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] *
                kernel, axis=(1, 2))

    return convolved_images
