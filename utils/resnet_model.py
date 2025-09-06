# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is a ResNet-50 model.

"""
import tensorflow as tf

def bottleneck_block(
    inputs, filters, strides, use_projection=False, data_format='channels_last'):
    """Bottleneck block variant for residual networks with BN after convolutions."""
    shortcut = inputs

    if use_projection:
        filters_out = 4 * filters
        shortcut = tf.keras.layers.Conv2D(
            filters=filters_out,
            kernel_size=1,
            strides=strides,
            padding='SAME',
            data_format=data_format)(inputs)
        shortcut = tf.keras.layers.BatchNormalization(
            axis=-1 if data_format == 'channels_last' else 1)(shortcut)

    bn1 = tf.keras.layers.BatchNormalization(
        axis=-1 if data_format == 'channels_last' else 1)(inputs)
    relu1 = tf.keras.layers.ReLU()(bn1)
    conv1 = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=1, strides=1, padding='SAME', data_format=data_format)(relu1)

    bn2 = tf.keras.layers.BatchNormalization(
        axis=-1 if data_format == 'channels_last' else 1)(conv1)
    relu2 = tf.keras.layers.ReLU()(bn2)
    conv2 = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, strides=strides, padding='SAME', data_format=data_format)(relu2)

    bn3 = tf.keras.layers.BatchNormalization(
        axis=-1 if data_format == 'channels_last' else 1)(conv2)
    relu3 = tf.keras.layers.ReLU()(bn3)
    conv3 = tf.keras.layers.Conv2D(
        filters=4 * filters, kernel_size=1, strides=1, padding='SAME', data_format=data_format)(relu3)

    return tf.keras.layers.add([conv3, shortcut])

def block_group(inputs, filters, blocks, strides, data_format='channels_last'):
    """Creates one group of blocks for the ResNet model."""
    net = bottleneck_block(inputs, filters, strides, use_projection=True, data_format=data_format)
    for _ in range(1, blocks):
        net = bottleneck_block(net, filters, 1, data_format=data_format)
    return net

def resnet_50(num_classes, data_format='channels_last', input_shape=(224, 224, 3)):
    """Returns the ResNet50 model for a given size and number of output classes."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=7, strides=2, padding='SAME', data_format=data_format)(inputs)
    x = tf.keras.layers.BatchNormalization(
        axis=-1 if data_format == 'channels_last' else 1)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=3, strides=2, padding='SAME', data_format=data_format)(x)

    x = block_group(x, 64, 3, 1, data_format)
    x = block_group(x, 128, 4, 2, data_format)
    x = block_group(x, 256, 6, 2, data_format)
    x = block_group(x, 512, 3, 2, data_format)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
