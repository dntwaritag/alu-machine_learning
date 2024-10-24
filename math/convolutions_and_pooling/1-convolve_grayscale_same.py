#!/usr/bin/env python3

'''This script performs a same convolution on grayscale images'''

import numpy as np


'''This function calculates the same convolution of grayscale images'''


def convolve_grayscale_same(images, kernel):
    '''
    Performs a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): Shape (m, h, w),
        containing multiple grayscale images.
        kernel (numpy.ndarray): Shape (kh, kw),
        containing the kernel for the convolution.

    Returns:
        numpy.ndarray: The convolved images.
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding
    pad_h = kh // 2
    pad_w = kw // 2

    # Apply padding
    padded_images = np.pad(
        images, ((0, 0),
                 (pad_h, pad_h),
                 (pad_w, pad_w)),
        mode='constant', constant_values=0)

    # Initialize output array
    convolved_images = np.zeros((m, h, w))

    # Perform convolution
    for i in range(h):
        for j in range(w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return convolved_images
