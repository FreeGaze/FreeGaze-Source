import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import random
import copy


class RGBSimpleModel(tf.keras.Model):
    """
    a simple deep neural networks for gaze estimation when input is RGB image
    """

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu',
                                             bias_initializer=tf.keras.initializers.constant(0.01),
                                             kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                             bias_regularizer=regularizers.l2(1e-4))
        self.conv2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, strides=2,activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv3 = tf.keras.layers.Conv2D(filters=72, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv4 = tf.keras.layers.Conv2D(filters=108, kernel_size=3, strides=2,activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv5 = tf.keras.layers.Conv2D(filters=162, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2,activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, bias_initializer=tf.keras.initializers.constant(0.01),
                                            activation=tf.nn.relu,
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(2, bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))

    def call(self, input):

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)

        return output


class DCTSimpleModel(tf.keras.Model):
    """
    base encoder for contrastive learning when input are DCT coefficients
    """

    def __init__(self):
        super().__init__()
        self.conv_y = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=2, activation='relu',
                                             bias_initializer=tf.keras.initializers.constant(0.01),
                                             kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                             bias_regularizer=regularizers.l2(1e-4))
        self.conv_cb_cr = tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu',
                                                 bias_initializer=tf.keras.initializers.constant(0.01),
                                                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                 bias_regularizer=regularizers.l2(1e-4))
        self.conv3 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv4 = tf.keras.layers.Conv2D(filters=72, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv5 = tf.keras.layers.Conv2D(filters=108, kernel_size=3, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv6 = tf.keras.layers.Conv2D(filters=162, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, bias_initializer=tf.keras.initializers.constant(0.01),
                                            activation=tf.nn.relu,
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(2, bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))

    def call(self, inputs):
        y = inputs[0] / 1024
        cb_cr = inputs[1] / 1024
        downsampled_y = self.conv_y(y)
        cb_cr = self.conv_cb_cr(cb_cr)
        x = tf.concat((downsampled_y, cb_cr), axis=3)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)

        return output


class RGBBaseEncoder(tf.keras.Model):
    """
    base encoder for contrastive learning when inputs are RGB images
    """

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv2 = tf.keras.layers.Conv2D(filters=48, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv3 = tf.keras.layers.Conv2D(filters=72, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv4 = tf.keras.layers.Conv2D(filters=108, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv5 = tf.keras.layers.Conv2D(filters=162, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='relu',
                                            bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=2)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input):

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        output = self.flatten(x)

        return output


class RGBProjectionHead(tf.keras.Model):
    """
    projection head for contrastive learning
    """

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(512, bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        #self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(128, bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))

    def call(self, input, training=None):
        x = self.dense1(input)
        #x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        output = self.dense2(x)

        return output
    
