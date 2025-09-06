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

"""WideResNet model.

This is a WideResNet model suitable for CIFAR-10.
"""

import tensorflow as tf

def wide_resnet_block(inputs, filters, strides, data_format='channels_last'):
    """Wide ResNet block."""
    shortcut = inputs
    if strides > 1 or inputs.shape[-1 if data_format == 'channels_last' else 1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, strides=strides,
            padding='SAME', data_format=data_format)(inputs)

    bn1 = tf.keras.layers.BatchNormalization(
        axis=-1 if data_format == 'channels_last' else 1)(inputs)
    relu1 = tf.keras.layers.ReLU()(bn1)
    conv1 = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, strides=strides,
        padding='SAME', data_format=data_format)(relu1)

    bn2 = tf.keras.layers.BatchNormalization(
        axis=-1 if data_format == 'channels_last' else 1)(conv1)
    relu2 = tf.keras.layers.ReLU()(bn2)
    conv2 = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, strides=1,
        padding='SAME', data_format=data_format)(relu2)

    return tf.keras.layers.add([conv2, shortcut])

def wide_resnet_group(inputs, filters, strides, num_blocks, data_format='channels_last'):
    """Wide ResNet group of blocks."""
    net = wide_resnet_block(inputs, filters, strides, data_format)
    for _ in range(1, num_blocks):
        net = wide_resnet_block(net, filters, 1, data_format)
    return net

def wide_resnet(
    num_classes, depth=28, width=10, data_format='channels_last', input_shape=(32, 32, 3)):
    """Creates a Wide ResNet model."""
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6

    k = width
    filters = [16, 16 * k, 32 * k, 64 * k]

    inputs = tf.keras.layers.Input(shape=input_shape)

    conv0 = tf.keras.layers.Conv2D(
        filters=filters[0], kernel_size=3, strides=1,
        padding='SAME', data_format=data_format)(inputs)

    group1 = wide_resnet_group(conv0, filters[1], 1, n, data_format)
    group2 = wide_resnet_group(group1, filters[2], 2, n, data_format)
    group3 = wide_resnet_group(group2, filters[3], 2, n, data_format)

    bn = tf.keras.layers.BatchNormalization(
        axis=-1 if data_format == 'channels_last' else 1)(group3)
    relu = tf.keras.layers.ReLU()(bn)

    pool = tf.keras.layers.GlobalAveragePooling2D()(relu)
    logits = tf.keras.layers.Dense(num_classes)(pool)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model
