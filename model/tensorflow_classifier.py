"""
CNN classifier implemented in TensorFlow.
"""

import tensorflow as tf

class Classifier(tf.keras.Model):
    # initialize and define all layers
    def __init__(self, image_dims, out_dim):
        # run base class initializer
        super(Classifier, self).__init__()

        # define convolution layers
        self.conv_1 = tf.keras.layers.Conv2D(
            32, 3, strides=1, activation='relu', padding='same',
            input_shape=tuple(image_dims))
        self.conv_2 = tf.keras.layers.Conv2D(
            64, 3, strides=2, activation='relu', padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(
            128, 3, strides=2, activation='relu', padding='same')
        self.conv_4 = tf.keras.layers.Conv2D(
            256, 3, strides=2, activation='relu', padding='same')

        # define flattening layers
        self.flat_1 = tf.keras.layers.Flatten()

        # define fully connected layers
        self.fc_1 = tf.keras.layers.Dense(out_dim)

    # compute forward propagation of input x
    def call(self, x, **kwargs):
        z_1 = self.conv_1(x)
        z_2 = self.conv_2(z_1)
        z_3 = self.conv_3(z_2)
        z_4 = self.conv_4(z_3)
        z_4_flat = self.flat_1(z_4)
        z_5 = self.fc_1(z_4_flat)
        return z_5
