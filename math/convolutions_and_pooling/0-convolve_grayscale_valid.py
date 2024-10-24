#!/usr/bin/env python3

'''This script performs a valid convolution on grayscale images'''

import numpy as np


'''This function calculates the valid convolution of grayscale images'''


def convolve_grayscale_valid(images, kernel):
    '''args:
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        kh is the height of the kernel
        kw is the width of the kernel
    '''
    m, h, w = images.shape
    kw, kh = kernel.shape
    new_h = h - kh + 1
    new_w = w - kw + 1
    convolved_images = np.zeros((m, new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            convolved_images[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
    return np.array(convolved_images)
