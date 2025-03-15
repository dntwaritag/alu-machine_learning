#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def flip_image(image):
    """
    Flips an image

    Param:
        - a 3d tf.tensor

    Returns:
        - flipped image
    """
    return tf.image.flip_left_right(image)


if __name__ == "__main__":
    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(flip_image(image))
        plt.show()
