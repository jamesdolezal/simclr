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
"""The main training pipeline."""

import json
import math
import os

from tqdm import tqdm
from slideflow import log as logging
from . import data as data_lib
from . import metrics
from . import model as model_lib
from . import objective as obj_lib
from . import utils as utils_lib

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# -----------------------------------------------------------------------------

def build_saved_model(
    model,
    include_projection_head=True,
    include_supervised_head=True
):
  """Returns a tf.Module for saving to SavedModel."""

  class SimCLRModel(tf.Module):
    """Saved model for exporting to hub."""

    def __init__(self, model):
      self.model = model
      # This can't be called `trainable_variables` because `tf.Module` has
      # a getter with the same name.
      self.trainable_variables_list = model.trainable_variables

    @tf.function
    def __call__(self, inputs, trainable):
      self.model(inputs, training=trainable)
      return utils_lib.get_salient_tensors_dict(
        include_projection_head, include_supervised_head
      )

  module = SimCLRModel(model)
  input_spec = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)
  module.__call__.get_concrete_function(input_spec, trainable=True)
  module.__call__.get_concrete_function(input_spec, trainable=False)
  return module


def save(model, destination, simclr_args, global_step=None, named_by_step=False):
  """Export as SavedModel for finetuning and inference."""
  is_supervised = ((simclr_args.train_mode == 'finetune'
                    or simclr_args.lineareval_while_pretraining)
                   and simclr_args.num_classes > 0)
  saved_model = build_saved_model(model, include_supervised_head=is_supervised)
  if named_by_step:
    checkpoint_export_dir = destination + f'_step{global_step}'
  else:
    checkpoint_export_dir = destination
  if tf.io.gfile.exists(checkpoint_export_dir):
    tf.io.gfile.rmtree(checkpoint_export_dir)
  tf.saved_model.save(saved_model, checkpoint_export_dir)
  with open(os.path.join(checkpoint_export_dir, 'args.json'), "w") as data_file:
    json.dump(simclr_args.to_dict(), data_file, indent=1)


def load(path, as_pretrained: bool = False):
    """Load a SavedModel or checkpoint for inference.

    Args:
        path (str): Path to saved model.

    Returns:
        Tensorflow SimCLR model.
    """
    args = utils_lib.load_model_args(path)
    if as_pretrained:
      args.train_mode = 'pretrain'
    model = model_lib.SimCLR(**args.model_kwargs)
    step = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(model=model, global_step=step)
    if path.endswith('.ckpt'):
        path = path.split('.ckpt')[0]
    checkpoint.restore(path).expect_partial()
    return model


def try_restore_from_checkpoint(
    model,
    global_step,
    optimizer,
    model_dir,
    checkpoint_path,
    keep_checkpoint_max=5,
    zero_init_logits_layer=False,
  ):
  """Restores the latest ckpt if it exists, otherwise check checkpoint_path"""
  checkpoint = tf.train.Checkpoint(
      model=model, global_step=global_step, optimizer=optimizer)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=model_dir,
      max_to_keep=keep_checkpoint_max)
  latest_ckpt = checkpoint_manager.latest_checkpoint
  if latest_ckpt:
    # Restore model weights, global step, optimizer states
    logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif checkpoint_path:
    # Restore model weights only, but not global step and optimizer states
    logging.info('Restoring from given checkpoint: %s', checkpoint_path)
    checkpoint_manager2 = tf.train.CheckpointManager(
        tf.train.Checkpoint(model=model),
        directory=model_dir,
        max_to_keep=keep_checkpoint_max)
    checkpoint_manager2.checkpoint.restore(checkpoint_path).expect_partial()
    if zero_init_logits_layer:
      model = checkpoint_manager2.checkpoint.model
      output_layer_parameters = model.supervised_head.trainable_weights
      logging.info('Initializing output layer parameters %s to zero',
                   [x.op.name for x in output_layer_parameters])
      for x in output_layer_parameters:
        x.assign(tf.zeros_like(x))

  return checkpoint_manager


def checkpoint_to_saved_model(ckpt, args, dest, global_step=0):
    model = model_lib.SimCLR(**args.model_kwargs)
    checkpoint = tf.train.Checkpoint(
        model=model,
        global_step=tf.Variable(0, dtype=tf.int64)
    )
    checkpoint.restore(ckpt).expect_partial()
    save(model, dest, args, global_step=global_step)

# -----------------------------------------------------------------------------

def perform_evaluation(
  model,
  builder,
  eval_steps,
  ckpt,
  strategy,
  model_dir,
  cache_dataset,
  args,
):
  """Perform evaluation."""
  if args.train_mode == 'pretrain' and not args.lineareval_while_pretraining:
    logging.info('Skipping eval during pretraining without linear eval.')
    return
  elif not builder.num_classes:
    logging.info('Skipping eval during pretraining; no labels supplied.')
  # Build input pipeline.
  ds = data_lib.build_distributed_dataset(
    builder, args.eval_batch_size, False, args, strategy,
    cache_dataset=cache_dataset
  )
  summary_writer = tf.summary.create_file_writer(model_dir)

  # Build metrics.
  with strategy.scope():
    regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
    label_top_1_accuracy = tf.keras.metrics.Accuracy(
        'eval/label_top_1_accuracy')
    label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
        5, 'eval/label_top_5_accuracy')
    all_metrics = [
        regularization_loss, label_top_1_accuracy, label_top_5_accuracy
    ]

    # Restore checkpoint.
    logging.info('Restoring from %s', ckpt)
    checkpoint = tf.train.Checkpoint(
        model=model, global_step=tf.Variable(0, dtype=tf.int64))
    checkpoint.restore(ckpt).expect_partial()
    global_step = checkpoint.global_step
    logging.info('Performing eval at step %d', global_step.numpy())

  def single_step(features, labels):
    _, supervised_head_outputs = model(features, training=False)
    assert supervised_head_outputs is not None
    outputs = supervised_head_outputs
    l = labels['labels']
    metrics.update_finetune_metrics_eval(label_top_1_accuracy,
                                         label_top_5_accuracy, outputs, l)
    reg_loss = model_lib.add_weight_decay(
        model, args.optimizer, args.weight_decay, adjust_per_optimizer=True
    )
    regularization_loss.update_state(reg_loss)

  with strategy.scope():

    @tf.function
    def run_single_step(iterator):
      images, labels = next(iterator)
      features, labels = images, {'labels': labels}
      strategy.run(single_step, (features, labels))

    iterator = iter(ds)
    for i in range(eval_steps):
      run_single_step(iterator)
      logging.info('Completed eval for %d / %d steps', i + 1, eval_steps)
    logging.info('Finished eval for %s', ckpt)

  # Write summaries
  cur_step = global_step.numpy()
  logging.info('Writing summaries for %d step', cur_step)
  with summary_writer.as_default():
    metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
    summary_writer.flush()

  # Record results as JSON.
  result_json_path = os.path.join(model_dir, 'result.json')
  result = {metric.name: metric.result().numpy() for metric in all_metrics}
  result['global_step'] = global_step.numpy()
  logging.info(result)
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  result_json_path = os.path.join(
      model_dir, 'result_%d.json'%result['global_step'])
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  flag_json_path = os.path.join(model_dir, 'args.json')
  with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    serializable_flags = {}

    for key, val in vars(args).items():
      # Some flag value types e.g. datetime.timedelta are not json serializable,
      # filter those out.
      if utils_lib.json_serializable(val):
        serializable_flags[key] = val
    json.dump(serializable_flags, f, indent=1)

  # Export as SavedModel for finetuning and inference.
  save(
    model,
    os.path.join(model_dir, 'saved_model'),
    simclr_args=args,
    global_step=result['global_step'],
    named_by_step=True
  )
  return result


def run_simclr(
  args,
  builder=None,
  model_dir=None,
  cache_dataset=False,
  checkpoint_path=None,
  use_tpu=False,
  tpu_name=None,
  tpu_zone=None,
  gcp_project=None,
):
  """Train a SimCLR model.

  Args:
    simCLR_args (SimpleNamespace): SimCLR arguments, as provided by
      :func:`slideflow.simclr.get_args`.
    builder (DatasetBuilder, optional): Builder for preparing SimCLR input
        pipelines. If None, will build using TensorflowDatasets and
        `simclr_args.dataset`.
    model_dir (str): Model directory for training.
    cache_dataset (bool): Whether to cache the entire dataset in memory. If
      the dataset is ImageNet, this is a very bad idea, but for smaller datasets
      it can improve performance
    checkpoint_path (str): Loading from the given checkpoint for fine-tuning if
      a finetuning checkpoint does not already exist in model_dir
    use_tpu (bool): Whether to run on TPU.
    tpu_name (str): The Cloud TPU to use for training. This should be either the
      name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470
      url
    tpu_zone (str): GCE zone where the Cloud TPU is located in. If not
      specified, we will attempt to automatically detect the GCE project from
      metadata
    gcp_project (str): Project name for the Cloud TPU-enabled project. If not
      specified, we will attempt to automatically detect the GCE project from
      metadata

  """
  logging.debug("Building SimCLR dataset")
  if builder is None:
    builder = tfds.builder(args.dataset, data_dir=args.data_dir)
    builder.download_and_prepare()
  num_train_examples = builder.info.splits[args.train_split].num_examples
  num_eval_examples = builder.info.splits[args.eval_split].num_examples
  args.num_classes = builder.info.features['label'].num_classes

  train_steps = model_lib.get_train_steps(num_train_examples, args.train_steps,
    args.train_epochs, args.train_batch_size)
  eval_steps = args.eval_steps or int(
      math.ceil(num_eval_examples / args.eval_batch_size))
  epoch_steps = int(round(num_train_examples / args.train_batch_size))

  logging.info(f"SimCLR Args: {json.dumps(args.to_dict(), indent=1)}")
  logging.info('# train examples: %d', num_train_examples)
  logging.info('# train_steps: %d', train_steps)
  logging.info('# eval examples: %d', num_eval_examples)
  logging.info('# eval steps: %d', eval_steps)

  checkpoint_steps = (
      args.checkpoint_steps or (args.checkpoint_epochs * epoch_steps))

  topology = None
  if use_tpu:
    logging.debug("Configuring TPUs")
    if tpu_name:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu_name, zone=tpu_zone, project=gcp_project)
    else:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(args.master)
    tf.config.experimental_connect_to_cluster(cluster)
    topology = tf.tpu.experimental.initialize_tpu_system(cluster)
    logging.info('Topology:')
    logging.info('num_tasks: %d', topology.num_tasks)
    logging.info('num_tpus_per_task: %d', topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(cluster)

  else:
    # For (multiple) GPUs.
    logging.debug("Configuring distributed dataset with MirroredStrategy")
    strategy = tf.distribute.MirroredStrategy()
    logging.info('Running using MirroredStrategy on %d replicas',
                 strategy.num_replicas_in_sync)

  with strategy.scope():
    model = model_lib.SimCLR(**args.model_kwargs)

  if args.mode == 'eval':
    logging.debug("Performing evaluation")
    for ckpt in tf.train.checkpoints_iterator(
        model_dir, min_interval_secs=15):
      result = perform_evaluation(
        model, builder, eval_steps, ckpt, strategy,
        model_dir, cache_dataset, args
      )
      if result['global_step'] >= train_steps:
        logging.info('Eval complete. Exiting...')
        return
  else:
    logging.debug("Setting up file writer for logs")
    summary_writer = tf.summary.create_file_writer(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'args.json'), "w") as data_file:
      json.dump(args.to_dict(), data_file, indent=1)
    with strategy.scope():
      # Build input pipeline.
      logging.debug("Setting up distributed dataset")
      ds = data_lib.build_distributed_dataset(builder, args.train_batch_size,
                                              True, args, strategy)

      # Build LR schedule and optimizer.
      learning_rate = model_lib.WarmUpAndCosineDecay(
        learning_rate=args.learning_rate,
        num_examples=num_train_examples,
        warmup_epochs=args.warmup_epochs,
        train_batch_size=args.train_batch_size,
        learning_rate_scaling=args.learning_rate_scaling,
        train_steps=args.train_steps,
        train_epochs=args.train_epochs
      )
      optimizer = model_lib.build_optimizer(
        learning_rate=learning_rate,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay
      )

      # Build metrics.
      all_metrics = []  # For summaries.
      weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
      total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
      all_metrics.extend([weight_decay_metric, total_loss_metric])
      if args.train_mode == 'pretrain':
        contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
        contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
        contrast_entropy_metric = tf.keras.metrics.Mean(
            'train/contrast_entropy')
        all_metrics.extend([
            contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric
        ])
      if args.train_mode == 'finetune' or args.lineareval_while_pretraining:
        supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
        supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
        all_metrics.extend([supervised_loss_metric, supervised_acc_metric])

      # Restore checkpoint if available.
      logging.debug("Attempting to restore from checkpoint")
      checkpoint_manager = try_restore_from_checkpoint(
          model, optimizer.iterations, optimizer, model_dir, checkpoint_path,
          keep_checkpoint_max=args.keep_checkpoint_max,
          zero_init_logits_layer=args.zero_init_logits_layer)

    steps_per_loop = min(checkpoint_steps, train_steps)

    def single_step(features, labels):
      with tf.GradientTape() as tape:
        # Log summaries on the last step of the training loop to match
        # logging frequency of other scalar summaries.
        #
        # Notes:
        # 1. Summary ops on TPUs get outside compiled so they do not affect
        #    performance.
        # 2. Summaries are recorded only on replica 0. So effectively this
        #    summary would be written once per host when should_record == True.
        # 3. optimizer.iterations is incremented in the call to apply_gradients.
        #    So we use  `iterations + 1` here so that the step number matches
        #    those of scalar summaries.
        # 4. We intentionally run the summary op before the actual model
        #    training so that it can run in parallel.
        should_record = tf.equal((optimizer.iterations + 1) % steps_per_loop, 0)
        with tf.summary.record_if(should_record):
          # Only log augmented images for the first tower.
          tf.summary.image(
              'image', features[:, :, :, :3], step=optimizer.iterations + 1)
        projection_head_outputs, supervised_head_outputs = model(
            features, training=True)
        loss = None
        if projection_head_outputs is not None:
          outputs = projection_head_outputs
          con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
              outputs,
              hidden_norm=args.hidden_norm,
              temperature=args.temperature,
              strategy=strategy)
          if loss is None:
            loss = con_loss
          else:
            loss += con_loss
          metrics.update_pretrain_metrics_train(contrast_loss_metric,
                                                contrast_acc_metric,
                                                contrast_entropy_metric,
                                                con_loss, logits_con,
                                                labels_con)
        if supervised_head_outputs is not None:
          outputs = supervised_head_outputs
          l = labels['labels']
          if (args.train_mode == 'pretrain'
              and args.lineareval_while_pretraining
              and args.num_classes):
            l = tf.concat([l, l], 0)
          sup_loss = obj_lib.add_supervised_loss(labels=l, logits=outputs)
          if loss is None:
            loss = sup_loss
          else:
            loss += sup_loss
          metrics.update_finetune_metrics_train(supervised_loss_metric,
                                                supervised_acc_metric, sup_loss,
                                                l, outputs)
        weight_decay = model_lib.add_weight_decay(
            model, args.optimizer, args.weight_decay, adjust_per_optimizer=True
        )
        weight_decay_metric.update_state(weight_decay)
        loss += weight_decay
        total_loss_metric.update_state(loss)
        # The default behavior of `apply_gradients` is to sum gradients from all
        # replicas so we divide the loss by the number of replicas so that the
        # mean gradient is applied.
        loss = loss / strategy.num_replicas_in_sync
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    with strategy.scope():

      @tf.function
      def train_single_step(iterator):
        # Drop the "while" prefix created by tf.while_loop which otherwise
        # gets prefixed to every variable name. This does not affect training
        # but does affect the checkpoint conversion script.
        # TODO(b/161712658): Remove this.
        with tf.name_scope(''):
          images, labels = next(iterator)
          features, labels = images, {'labels': labels}
          strategy.run(single_step, (features, labels))

      def train_multiple_steps(iterator):
        for _ in tqdm(range(steps_per_loop)):
          train_single_step(iterator)

      global_step = optimizer.iterations
      cur_step = global_step.numpy()
      iterator = iter(ds)
      logging.debug("Beginning training")
      while cur_step < train_steps:
        # Calls to tf.summary.xyz lookup the summary writer resource which is
        # set by the summary writer's context manager.
        with summary_writer.as_default():
          train_multiple_steps(iterator)
          cur_step = global_step.numpy()
          checkpoint_manager.save(cur_step)
          logging.info('Completed: %d / %d steps', cur_step, train_steps)
          metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
          tf.summary.scalar(
              'learning_rate',
              learning_rate(tf.cast(global_step, dtype=tf.float32)),
              global_step)
          summary_writer.flush()
        for metric in all_metrics:
          metric.reset_states()
      logging.info('Training complete...')

    if args.mode == 'train_then_eval':
      perform_evaluation(model, builder, eval_steps,
                         checkpoint_manager.latest_checkpoint, strategy,
                         model_dir, cache_dataset, args)
    else:
      # Export as SavedModel for finetuning and inference.
      save(
        model,
        os.path.join(model_dir, 'saved_model'),
        args,
        global_step=global_step)
