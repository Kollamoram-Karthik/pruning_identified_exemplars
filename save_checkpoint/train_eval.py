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

r"""Training script to sparsify a ResNet-50.

"""
import os
from absl import app
from absl import flags
import tensorflow as tf
from tqdm import tqdm



# model params
flags.DEFINE_enum('dataset', 'imagenet', ['imagenet', 'cifar10'], 'Dataset to use.')
flags.DEFINE_enum('model_name', 'resnet_50', ['resnet_50', 'wide_resnet'], 'Model to use.')
flags.DEFINE_integer('widenet_depth', 28, 'Depth of WideResNet.')
flags.DEFINE_integer('widenet_width', 10, 'Width of WideResNet.')
flags.DEFINE_integer(
    'steps_per_checkpoint', 500,
    'Controls how often checkpoints are generated. More steps per '
    'checkpoint = higher utilization of TPU and generally higher '
    'steps/sec')
flags.DEFINE_float('label_smoothing', 0.1,
                   'Relax confidence in the labels by (1-label_smoothing).')
flags.DEFINE_integer('steps_per_eval', 1251,
                     'Controls how often evaluation is performed.')
flags.DEFINE_integer('num_cores', 8, 'Number of cores.')
flags.DEFINE_string('output_dir', '',
                    'Directory where to write event logs and checkpoint.')
flags.DEFINE_string('mode', 'train',
                    'One of {"train_and_eval", "train", "eval"}.')
flags.DEFINE_string('train_dir', '',
                    'The location of the tfrecords used for training.')
flags.DEFINE_string('eval_dir', '',
                    'The location of the tfrecords used for eval.')


# pruning flags
flags.DEFINE_string('pruning_hparams', '',
                    'Comma separated list of pruning-related hyperparameters')
flags.DEFINE_float('end_sparsity', 0.1,
                   'Target sparsity desired by end of training.')
flags.DEFINE_integer('sparsity_begin_step', 5000, 'Step to begin pruning at.')
flags.DEFINE_integer('sparsity_end_step', 8000, 'Step to end pruning at.')
flags.DEFINE_integer('pruning_frequency', 500, 'Step interval between pruning.')
flags.DEFINE_enum(
    'pruning_method', 'baseline',
    ('threshold', 'random_independent', 'random_cumulative', 'baseline'),
    'Method used for pruning'
    'Specify as baseline if no pruning is used.')
flags.DEFINE_bool('log_class_level_summaries', True,
                  'Boolean for whether to log class level precision/accuracy.')
flags.DEFINE_float('expansion_factor', 6.,
                   'how much to expand filters before depthwise conv')
flags.DEFINE_float(
    'training_steps_multiplier', 1.0,
    'Training schedule is shortened or extended with the '
    'multiplier, if it is not 1.')
flags.DEFINE_integer('block_width', 1, 'width of block')
flags.DEFINE_integer('block_height', 1, 'height of block')

# set this flag to true to do a test run of this code with synthetic data
flags.DEFINE_bool('test_small_sample', True,
                  'Boolean for whether to test internally.')

FLAGS = flags.FLAGS

imagenet_params = {
    'sloppy_shuffle': True,
    'num_cores': 8,
    'train_batch_size': 4096,
    'num_train_images': 1281167,
    'num_eval_images': 50000,
    'num_label_classes': 1000,
    'num_train_steps': 32000,
    'base_learning_rate': 0.1,
    'weight_decay': 1e-4,
    'eval_batch_size': 1024,
    'mean_rgb': [0.485 * 255, 0.456 * 255, 0.406 * 255],
    'stddev_rgb': [0.229 * 255, 0.224 * 255, 0.225 * 255]
}

cifar10_params = {
    'sloppy_shuffle': True,
    'num_cores': 8,
    'train_batch_size': 128,
    'num_train_images': 50000,
    'num_eval_images': 10000,
    'num_label_classes': 10,
    'num_train_steps': 20000,
    'base_learning_rate': 0.1,
    'weight_decay': 2e-4,
    'eval_batch_size': 100,
    'mean_rgb': [0.4914 * 255, 0.4822 * 255, 0.4465 * 255],
    'stddev_rgb': [0.2023 * 255, 0.1994 * 255, 0.2010 * 255]
}


from pruning_identified_exemplars.utils import data_input
from pruning_identified_exemplars.utils import data_input_cifar10
from pruning_identified_exemplars.utils import resnet_model
from pruning_identified_exemplars.utils import widenet_model

def main(argv):
  del argv  # Unused.

  if FLAGS.dataset == 'imagenet':
      params = imagenet_params
      input_fn = data_input.input_fn
  elif FLAGS.dataset == 'cifar10':
      params = cifar10_params
      input_fn = data_input_cifar10.input_fn
  else:
      raise ValueError('Invalid dataset name: {}'.format(FLAGS.dataset))

  params['mode'] = FLAGS.mode
  if FLAGS.mode == 'train':
    params['batch_size'] = params['train_batch_size']
  else:
    params['batch_size'] = params['eval_batch_size']

  if FLAGS.model_name == 'resnet_50':
      model = resnet_model.resnet_50(
          num_classes=params["num_label_classes"],
          pruning_method=FLAGS.pruning_method,
          data_format="channels_last")
  elif FLAGS.model_name == 'wide_resnet':
      model = widenet_model.wide_resnet(
          num_classes=params["num_label_classes"],
          depth=FLAGS.widenet_depth,
          width=FLAGS.widenet_width,
          data_format="channels_last")
  else:
      raise ValueError('Invalid model name: {}'.format(FLAGS.model_name))

  optimizer = tf.keras.optimizers.SGD(learning_rate=params['base_learning_rate'], momentum=0.9)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  train_dataset = input_fn(params)

  if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

  summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.output_dir, 'logs'))
  steps_per_epoch = params['num_train_images'] // params['train_batch_size']
  train_iterator = iter(train_dataset)
  with tqdm(range(params['num_train_steps']), desc="Training") as pbar:
    for step in pbar:
        if step % steps_per_epoch == 0:
            pbar.set_description('Epoch {}'.format(step // steps_per_epoch))
        try:
            x_batch_train, y_batch_train = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataset)
            x_batch_train, y_batch_train = next(train_iterator)

        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        pbar.set_postfix(loss=loss_value.numpy())

        with summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=step)

        if step % 100 == 0:
            with summary_writer.as_default():
                for var in model.trainable_variables:
                    tf.summary.histogram(var.name, var, step=step)

        if step % FLAGS.steps_per_checkpoint == 0:
            checkpoint.save(file_prefix=os.path.join(FLAGS.output_dir, 'ckpt'))


if __name__ == '__main__':
  app.run(main)
