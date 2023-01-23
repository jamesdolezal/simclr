# coding=utf-8
# Copyright 2020 The SimCLR Authors.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data pipeline."""

import functools
from slideflow import log as logging

from . import data_util
from .data_util import FLAGS
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class SlideflowBuilder:

    def __init__(self, train_dts=None, val_dts=None, test_dts=None, labels=None,
                 num_classes=None, val_kwargs=None, steps_per_epoch_override=None,
                 dataset_kwargs=None):
        """Build a training/validet."""
        if train_dts is None and val_dts is None and test_dts is None:
            raise ValueError("Must supply either train_dts, val_dts, or test_dts.")
        if labels is not None and num_classes is None:
            raise ValueError("If labels is not None, must specify `num_classes`")
        if val_kwargs is not None and val_dts is not None:
            raise ValueError("Cannot supply val_kwargs if val_dts is not None")
        if val_kwargs is not None and train_dts is None:
            raise ValueError("Cannot supply val_kwargs if train_dts is None")

        self.labels = labels
        if val_kwargs is not None:
            self.train_dts, self.val_dts = train_dts.train_val_split(
                labels=self.labels,
                **val_kwargs
            )
        else:
            self.train_dts = train_dts
            self.val_dts = val_dts
            self.test_dts = test_dts
        if steps_per_epoch_override:
            train_tiles = steps_per_epoch_override
        elif self.train_dts:
            train_tiles = self.train_dts.num_tiles
        else:
            train_tiles = 0
        self.dataset_kwargs = dict() if dataset_kwargs is None else dataset_kwargs
        self.info = data_util.EasyDict(
            features=data_util.EasyDict(
                label=data_util.EasyDict(num_classes=num_classes)
            ),
            splits=data_util.EasyDict(
                train=data_util.EasyDict(num_examples=train_tiles),
                validation=data_util.EasyDict(num_examples=(0 if not self.val_dts else self.val_dts.num_tiles)),
                test=data_util.EasyDict(num_examples=(0 if not self.test_dts else self.test_dts.num_tiles))
            ))

    def as_dataset(self, split, read_config, shuffle_files, as_supervised):
        logging.info(f"Dataset split requested: {split}")
        if split == 'train':
            dts = self.train_dts
        elif split == 'validation':
            dts = self.val_dts
        elif split == 'test':
            dts = self.test_dts
        else:
            raise ValueError(f"Unrecognized split {split}, expected 'train' "
                             "'validation', or 'test'.")
        if dts is None:
            raise ValueError(f'Builder not configured for phase "{split}".')

        return dts.tensorflow(
            labels=self.labels,
            num_shards=read_config.input_context.num_input_pipelines,
            shard_idx=read_config.input_context.input_pipeline_id,
            deterministic=True,
            standardize=False,
            infinite=(split == 'train'),
            **self.dataset_kwargs
        )


def build_input_fn(builder, global_batch_size, topology, is_training):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label, *args):
      """Produces multiple transformations of the same batch."""
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      if num_classes:
        label = tf.one_hot(label, num_classes)
      return image, label, *args

    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training,
        as_supervised=True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1,
            input_context=input_context))
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(builder, batch_size, is_training, strategy,
                              topology):
  input_fn = build_input_fn(builder, batch_size, topology, is_training)
  return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if FLAGS.image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  return functools.partial(
      data_util.preprocess_image,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=test_crop)
