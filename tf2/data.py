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
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class DatasetBuilder:

    def __init__(self, train_dts=None, val_dts=None, test_dts=None, *, labels=None,
                 val_kwargs=None, steps_per_epoch_override=None,
                 dataset_kwargs=None):
        """Build a training/validation dataset pipeline for SimCLR.

        Args:
            train_dts (sf.Dataset, optional): Training dataset.
            val_dts (sf.Dataset, optional): Optional validation dataset.
            test_dts (sf.Dataset, optional): Optional held-out test set.

        Keyword args:
            labels (str or dict): Labels for training the supervised head.
                Can be a name of an outcome (str) or a dict mapping slide names
                to labels.
            val_kwargs (dict, optional): Optional keyword arguments for
                generating a validation dataset from ``train_dts`` via
                ``train_dts.split()``. Incompatible with ``val_dts``.
            steps_per_epoch_override (int, optional): Override the number
                of steps per epoch.
            dataset_kwargs (dict, optional): Keyword arguments passed to the
                :meth:`slideflow.Dataset.tensorflow` method when creating
                the input pipeline.

        """
        if train_dts is None and val_dts is None and test_dts is None:
            raise ValueError("Must supply either train_dts, val_dts, or test_dts.")
        if val_kwargs is not None and val_dts is not None:
            raise ValueError("Cannot supply val_kwargs if val_dts is not None")
        if val_kwargs is not None and train_dts is None:
            raise ValueError("Cannot supply val_kwargs if train_dts is None")

        if isinstance(labels, dict):
            self.labels = labels
        elif isinstance(labels, str):
            self.labels = {}
            if train_dts is not None:
                self.labels.update(train_dts.labels(labels)[0])
            if val_dts is not None:
                self.labels.update(val_dts.labels(labels)[0])
            if test_dts is not None:
                self.labels.update(test_dts.labels(labels)[0])
        elif labels is not None:
            raise ValueError(
                f"Unrecognized type {type(labels)} for argument labels: "
                "expected dict or str"
            )
        else:
            self.labels = None
        if val_kwargs is not None:
            if self.labels is None:
                raise ValueError(
                    "Unable to automatically generate training/validation "
                    "splits using keyword arguments (val_kwargs) "
                    "if labels are not provided."
                )
            self.train_dts, self.val_dts = train_dts.split(
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
        self.num_classes = 0 if self.labels is None else len(set(list(self.labels.values())))
        self.dataset_kwargs = dict() if dataset_kwargs is None else dataset_kwargs
        self.info = data_util.EasyDict(
            features=data_util.EasyDict(
                label=data_util.EasyDict(num_classes=self.num_classes)
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

    def build_dataset(self, *args, **kwargs):
        """Builds a distributed dataset.

        Args:
            batch_size (int): Global batch size across devices.
            is_training (bool): If this is for training.
            simclr_args (SimCLR_Args): SimCLR arguments.
            strategy (tf.distribute.Strategy, optional): Distribution strategy.
            cache_dataset (bool): Cache dataset.

        Returns:
            Distributed Tensorflow dataset, with SimCLR preprocessing applied.
        """
        return build_distributed_dataset(self, *args, **kwargs)


def build_input_fn(builder, global_batch_size, is_training,
                   simclr_args, cache_dataset=False):
  """Build input function.

  Args:
    builder: Either DatasetBuilder, or a TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    is_training: Whether to build in training mode.
    simCLR_args:  SimCLR arguments, as provided by :func:`slideflow.simclr.get_args`.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(
      is_training, is_pretrain=True, image_size=simclr_args.image_size,
      color_jitter_strength=simclr_args.color_jitter_strength)
    preprocess_fn_finetune = get_preprocess_fn(
      is_training, is_pretrain=False, image_size=simclr_args.image_size,
      color_jitter_strength=simclr_args.color_jitter_strength)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label, *args):
      """Produces multiple transformations of the same batch."""
      if is_training and simclr_args.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      if num_classes:
        label = tf.one_hot(label, num_classes)
      return detuple(image, label, args)

    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    dataset = builder.as_dataset(
        split=simclr_args.train_split if is_training else simclr_args.eval_split,
        shuffle_files=is_training,
        as_supervised=True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1,
            input_context=input_context))
    if cache_dataset:
      dataset = dataset.cache()
    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      buffer_multiplier = 50 if simclr_args.image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(builder, batch_size, is_training, simclr_args,
                              strategy=None, cache_dataset=False):
  if strategy is None:
    strategy = tf.distribute.get_strategy()
  input_fn = build_input_fn(
    builder, batch_size, is_training, simclr_args, cache_dataset=cache_dataset
  )
  return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, is_pretrain, image_size, color_jitter_strength=1.0):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  return functools.partial(
      data_util.preprocess_image,
      height=image_size,
      width=image_size,
      color_jitter_strength=color_jitter_strength,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=test_crop)

# -----------------------------------------------------------------------------

def detuple(image, label, args):
    """Detuple optional arguments for return.

    Adds support for returning args via wildcard in Python 3.7. The following:

    .. code-block:: python

        return image, label, *args

    can be made cross-compatible with Python 3.7 and higher by using:

    .. code-block:: python

        return detuple(image, label, args)

    """
    if len(args):
        return tuple([image, label] + list(args))
    else:
        return image, label