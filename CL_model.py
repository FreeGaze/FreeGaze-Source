import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import random
import copy


class BaseEncoder(tf.keras.Model):
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

        output = self.flatten(x)

        return output


class ProjectionHead(tf.keras.Model):
    """
    projection head for contrastive learning
    """

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(512, bias_initializer=tf.keras.initializers.constant(0.01),
                                            activation=tf.nn.relu,
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.dense2 = tf.keras.layers.Dense(64, bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))

    def call(self, input):
        x = self.dense1(input)
        output = self.dense2(x)

        return output


class GazeEstimationHead(tf.keras.Model):
    """
    head for gaze estimation in contrastive learning
    """

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(512, bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(2, bias_initializer=tf.keras.initializers.constant(0.01),
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4))

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.bn1(x, training)
        x = tf.nn.relu(x)
        output = self.dense2(x)

        return output


class Resnet18_RGBBaseEncoder(tf.keras.Model):

    def __init__(self, layer_params=None):
        super(Resnet18_RGBBaseEncoder, self).__init__()

        if layer_params is None:
            layer_params = [2, 2, 2, 2]

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.flatten(x)

        return output
    

class Resnet18_BaseEncoder(tf.keras.Model):

    def __init__(self, layer_params=None):
        super(Resnet18_BaseEncoder, self).__init__()

        if layer_params is None:
            layer_params = [2, 2, 2, 2]
        self.conv_y = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu',
                                             bias_initializer=tf.keras.initializers.constant(0.01),
                                             kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                             bias_regularizer=regularizers.l2(1e-4))
        self.conv_cb_cr = tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation='relu',
                                                 bias_initializer=tf.keras.initializers.constant(0.01),
                                                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                 bias_regularizer=regularizers.l2(1e-4))
        #self.layer1 = make_basic_block_layer(filter_num=64,
                                             #blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        y = inputs[0] / 1024
        cb_cr = inputs[1] / 1024
        downsampled_y = self.conv_y(y)
        cb_cr = self.conv_cb_cr(cb_cr)
        x = tf.concat((downsampled_y, cb_cr), axis=3)
        #x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        output = self.flatten(x)

        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output
