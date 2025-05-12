#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer class
        """
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3:
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
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new, w_new = 512, int(w * 512 / h)
        else:
            h_new, w_new = int(h * 512 / w), 512

        scaled = tf.image.resize_bicubic(np.expand_dims(image, 0), (h_new, w_new))
        scaled = tf.clip_by_value(scaled / 255, 0, 1)
        return scaled

    def load_model(self):
        """
        Creates the model used to calculate cost
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        self.model = tf.keras.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates gram matrices
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or input_layer.ndim != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        """
        vgg19 = tf.keras.applications.vgg19
        preprocessed_style = vgg19.preprocess_input(self.style_image * 255)
        preprocessed_content = vgg19.preprocess_input(self.content_image * 255)
        style_features = self.model(preprocessed_style)
        content_features = self.model(preprocessed_content)
        self.gram_style_features = [self.gram_matrix(style_feature)
                                    for style_feature in style_features[:-1]]
        self.content_feature = content_features[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           style_output.ndim != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        c = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(c, c))
        
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image
        """
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != len(self.style_layers):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    len(self.style_layers)))
        
        total_style_cost = 0
        weight = 1 / len(self.style_layers)
        for style_output, gram_target in zip(style_outputs, self.gram_style_features):
            layer_style_cost = self.layer_style_cost(style_output, gram_target)
            total_style_cost += weight * layer_style_cost
        
        return total_style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for generated image
        """
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != self.content_feature.shape:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape))
        
        return tf.reduce_mean(tf.square(content_output - self.content_feature))
