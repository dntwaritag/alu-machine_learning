#!/usr/bin/env python3
"""
   function def shear_image(image, intensity):
   that randomly shears an image:
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def shear_image(image, intensity):
    """
    randomly shears an image

    Params:
        - image: 3d tf.tensor
        - intensity of the shear

    Returns:
        - sheared image
    """
    return tf.keras.preprocessing.image.random_shear(image, intensity=intensity)


if __name__ == "__main__":
    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(shear_image(image, 50))
        plt.show()
