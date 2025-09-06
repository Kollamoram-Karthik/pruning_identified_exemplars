
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

"""Helper functions to pre-process cifar10 data inputs and create data iterator.
"""

import tensorflow.compat.v1 as tf
from pruning_identified_exemplars.utils import preprocessing_helper

def input_fn(params):
  """Input function for CIFAR-10 dataset."""
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  if params['mode'] == 'train':
    x, y = x_train, y_train
  else:
    x, y = x_test, y_test

  def parser(x, y):
    image = tf.cast(x, tf.float32)
    if params['mode'] == 'train':
      image = preprocessing_helper.preprocess_image(
          image=image, image_size=32, is_training=True)
    else:
      image = preprocessing_helper.preprocess_image(
          image=image, image_size=32, is_training=False)
    return image, y

  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  if params['mode'] == 'train':
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()

  dataset = dataset.map(parser, num_parallel_calls=64)
  dataset = dataset.batch(params['batch_size'], drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
