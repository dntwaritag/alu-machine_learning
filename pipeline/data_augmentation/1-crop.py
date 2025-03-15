#!/usr/bin/env python3
'''
    function def crop_image(image, size):
    that performs a random crop of an image:
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def crop_image(image, size):
    """
    Performs a random crop of an image

    Params:
        - image: a 3d tf.tensor
        - size - a tuple of the size of the crop

    Returns:
        - Cropped image
    """
    return tf.image.random_crop(image, size=size)


if __name__ == "__main__":
    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(crop_image(image, (200, 200, 3)))
        plt.show()
