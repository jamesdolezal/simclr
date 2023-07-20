"""Utility functions."""

import json
import tensorflow as tf
import json
from os.path import dirname, join, exists
from slideflow import log

# -----------------------------------------------------------------------------

class SimCLR_Args:
    def __init__(
        self,
        learning_rate=0.075,
        learning_rate_scaling='sqrt',
        warmup_epochs=10,
        weight_decay=1e-4,
        batch_norm_decay=0.9,
        train_batch_size=512,
        train_split='train',
        train_epochs=100,
        train_steps=0,
        eval_steps=0,
        eval_batch_size=256,
        checkpoint_epochs=1,
        checkpoint_steps=0,
        eval_split='validation',
        dataset='imagenet2012',
        mode='train',
        train_mode='pretrain',
        lineareval_while_pretraining=True,
        zero_init_logits_layer=False,
        fine_tune_after_block=-1,
        master=None,
        data_dir=None,
        optimizer='lars',
        momentum=0.9,
        keep_checkpoint_max=5,
        temperature=0.1,
        hidden_norm=True,
        proj_head_mode='nonlinear',
        proj_out_dim=128,
        num_proj_layers=3,
        ft_proj_selector=0,
        global_bn=True,
        width_multiplier=1,
        resnet_depth=50,
        sk_ratio=0.,
        se_ratio=0.,
        image_size=224,
        color_jitter_strength=1.0,
        use_blur=True,
        num_classes=None,
        stain_augment=True,
    ) -> None:
      """SimCLR arguments.

      A class containg all default - if not overwritten at initialization -
        SimCLR arguments.

      Keyword Args:
        learning_rate (float): Initial learning rate per batch size of 256.
        learning_rate_scaling (str): How to scale the learning rate as a
          function of batch size. 'linear' or 'sqrt'.
        warmup_epochs (int): Number of epochs of warmup.
        weight_decay (float):  Amount of weight decay to use.
        batch_norm_decay (float): Batch norm decay parameter.
        train_batch_size (int): Batch size for training.
        train_split (str): Split for training
        train_epoch (int): Number of epochs to train for.
        train_step (int): Number of steps to train for. If provided, overrides
          train_epochs.
        eval_steps (int): Number of steps to eval for. If not provided, evals
          over entire dataset.
        eval_batch_size (int): Batch size for eval.
        checkpoint_epochs (int): Number of epochs between
          checkpoints/summaries.
        checkpoint_steps (int): Number of steps between checkpoints/summaries.
          If provided, overrides checkpoint_epochs.
        eval_split (str): Split for evaluation.
        dataset (str): Name of a dataset.
        mode (str): Whether to perform training or evaluation. 'train',
          'eval', or 'train_then_eval'
        train_mode (str): The train mode controls different objectives and
          trainable components.
        lineareval_while_pretraining (bool): Whether to finetune supervised
          head while pretraining. 'pretrain' or 'finetune'
        zero_init_logits_layer (bool): If True, zero initialize layers after
          avg_pool for supervised learning.
        fine_tune_after_block (int): The layers after which block that we will
            fine-tune. -1 means fine-tuning everything. 0 means fine-tuning
            after stem block. 4 means fine-tuning just the linear head.
        master (str): Address/name of the TensorFlow master to use.
          By default, use an in-process master.
        data_dir (str): Directory where dataset is stored.
        optimizer (str): Optimizer to use. 'momentum', 'adam', 'lars'
        momentum (float): Momentum parameter.
        keep_checkpoint_max (int): Maximum number of checkpoints to keep.
        temperature (float): Temperature parameter for contrastive loss.
        hidden_norm (bool): Temperature parameter for contrastive loss.
        proj_head_mode (str): How the head projection is done. 'none',
          'linear', 'nonlinear'
        proj_out_dim (int): Number of head projection dimension.
        num_proj_layers (int): Number of non-linear head layers.
        ft_proj_selector (int): Which layer of the projection head to use
          during fine-tuning. 0 means no projection head, and -1 means the
          final layer.
        global_bn (bool): Whether to aggregate BN statistics across
          distributed cores.
        width_multiplier (int): Multiplier to change width of network.
        resnet_depth (int): Depth of ResNet.
        sk_ratio (float): If it is bigger than 0, it will enable SK.
          Recommendation: 0.0625.
        se_ratio (float): If it is bigger than 0, it will enable SE.
        image_size (int): Input image size.
        color_jitter_strength (float): The strength of color jittering.
        use_blur (bool): Whether or not to use Gaussian blur for augmentation
          during pretraining.
        num_classes (int):  Number of classes for the supervised head.
      """
      for argname, argval in dict(locals()).items():
          setattr(self, argname, argval)

    def to_dict(self):
        return {k:v for k,v in vars(self).items()
                if k not in ('model_kwargs', 'self')}

    def __repr__(self):
        return '{}(\n{}\n)'.format(
            self.__class__.__name__,
            ',\n'.join('  {}={!r}'.format(k, v) for k, v in self.to_dict().items())
        )

    @property
    def model_kwargs(self):
        return {
            k: getattr(self, k)
            for k in ('num_classes', 'resnet_depth', 'width_multiplier',
                      'sk_ratio', 'se_ratio', 'image_size', 'batch_norm_decay',
                      'train_mode', 'use_blur', 'proj_out_dim', 'proj_head_mode',
                      'lineareval_while_pretraining', 'fine_tune_after_block',
                      'num_proj_layers', 'ft_proj_selector')
        }

# -----------------------------------------------------------------------------

def get_args(**kwargs):
    """Configure a ``SimCLR_Args`` object for training SimCLR.

    Keyword args:
        **kwargs: Please see the :class:`slideflow.simclr.SimCLR_Args` documentation
            for information on available parameters.

    Returns:
        slideflow.simclr.SimCLR_Args

    """
    return SimCLR_Args(**kwargs)


def load_model_args(model_path, ignore_missing=False):
    """Load args.json associated with a given SimCLR model or checkpoint.

    Args:
        model_path (str): Path to SimCLR model or checkpoint.

    Returns:
        Dictionary of contents of args.json file. If file is not found and
        `ignore_missing` is False, will return None. If `ignore_missing` is
        True, will raise an OSError.

    Raises:
        OSError: If args.json cannot be found and `ignore_missing` is False.
    """
    for flag_path in (join(model_path, 'args.json'),
                      join(dirname(model_path), 'args.json')):
        if exists(flag_path):
            with open(flag_path, 'r') as f:
                return SimCLR_Args(**json.load(f))
    if ignore_missing:
        return None
    else:
        raise OSError(f"Unable to find args.json for SimCLR model {model_path}")

# -----------------------------------------------------------------------------

def json_serializable(val):
  try:
    json.dumps(val)
    return True
  except TypeError:
    return False


def get_salient_tensors_dict(include_projection_head, include_supervised_head):
  """Returns a dictionary of tensors."""
  graph = tf.compat.v1.get_default_graph()
  result = {}
  for i in range(1, 5):
    result['block_group%d' % i] = graph.get_tensor_by_name(
        'resnet/block_group%d/block_group%d:0' % (i, i))
  result['initial_conv'] = graph.get_tensor_by_name(
      'resnet/initial_conv/Identity:0')
  result['initial_max_pool'] = graph.get_tensor_by_name(
      'resnet/initial_max_pool/Identity:0')
  result['final_avg_pool'] = graph.get_tensor_by_name('resnet/final_avg_pool:0')
  if include_supervised_head:
    result['logits_sup'] = graph.get_tensor_by_name(
        'head_supervised/logits_sup:0')
  if include_projection_head:
    result['proj_head_input'] = graph.get_tensor_by_name(
        'projection_head/proj_head_input:0')
    result['proj_head_output'] = graph.get_tensor_by_name(
        'projection_head/proj_head_output:0')
  return result

def _restore_latest_or_from_pretrain(checkpoint_manager, args, checkpoint_path):
  """Restores the latest ckpt if training already.

  Or restores from checkpoint_path if in finetune mode.

  Args:
    checkpoint_manager: tf.traiin.CheckpointManager.
  """
  latest_ckpt = checkpoint_manager.latest_checkpoint
  if latest_ckpt:
    # The model is not build yet so some variables may not be available in
    # the object graph. Those are lazily initialized. To suppress the warning
    # in that case we specify `expect_partial`.
    log.info('Restoring from %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif args.train_mode == 'finetune':
    # Restore from pretrain checkpoint.
    assert checkpoint_path, 'Missing pretrain checkpoint.'
    log.info('Restoring from %s', checkpoint_path)
    checkpoint_manager.checkpoint.restore(checkpoint_path).expect_partial()
    # TODO(iamtingchen): Can we instead use a zeros initializer for the
    # supervised head?
    if args.zero_init_logits_layer:
      model = checkpoint_manager.checkpoint.model
      output_layer_parameters = model.supervised_head.trainable_weights
      log.info('Initializing output layer parameters %s to zero',
                   [x.op.name for x in output_layer_parameters])
      for x in output_layer_parameters:
        x.assign(tf.zeros_like(x))
