#!/usr/bin/env python3

'''This script performs a same convolution on grayscale images'''

import numpy as np


'''This function calculates the same convolution of grayscale images'''


def convolve_grayscale(images, kernel,
                       padding='same', stride=(1, 1)):
    '''
    Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Shape (m, h, w),
        containing multiple grayscale images.
        kernel (numpy.ndarray): Shape (kh, kw),
        containing the kernel for the convolution.
        padding (str or tuple): Either a tuple of (ph, pw),
        'same', or 'valid'.
        stride (tuple): (sh, sw),
        containing the stride for the height and width of the image.

    Returns:
        numpy.ndarray: The convolved images.
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    # Apply padding
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)),
        mode='constant', constant_values=0)

    # Calculate dimensions of the output image
    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1

    # Initialize output array
    convolved_images = np.zeros((m, new_h, new_w))

    # Perform convolution
    for i in range(new_h):
        for j in range(new_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh+kh,
                              j*sw:j*sw+kw] *
                kernel, axis=(1, 2))

    return convolved_images
