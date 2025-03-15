#!/usr/bin/env python3
"""
   function def change_brightness(image, max_delta):
   that randomly changes the brightness of an image
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    Param:
        - image: 3d tf.tensor
        - max_delta is the mac amount the image should be brightened to

    Returns:
        - An altered image
    """
    return tf.image.random_brightness(image, max_delta=max_delta)


if __name__ == "__main__":
    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(change_brightness(image, 0.3))
        plt.show()
