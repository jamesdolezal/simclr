"""Utility functions."""

import json
from os.path import dirname, join, exists

# -----------------------------------------------------------------------------

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
                      join(dirname(model_path, 'args.json'))):
        if exists(flag_path):
            with open(flag_path, 'r') as f:
                return SimCLR_Args(**json.load(f))
    if ignore_missing:
        return None
    else:
        raise OSError(f"Unable to find args.json for SimCLR model {model_path}")


def get_args(**kwargs):
    return SimCLR_Args(**kwargs)


class SimCLR_Args:
    def __init__(
        self,
        learning_rate=0.3,                   # Initial learning rate per batch size of 256.
        learning_rate_scaling='linear',      # How to scale the learning rate as a function of batch size. 'linear' or 'sqrt'
        warmup_epochs=10,                    # Number of epochs of warmup.
        weight_decay=1e-6,                   # Amount of weight decay to use.
        batch_norm_decay=0.9,                # Batch norm decay parameter.
        train_batch_size=512,                # Batch size for training.
        train_split='train',                 # Split for training.
        train_epochs=100,                    # Number of epochs to train for.
        train_steps=0,                       # Number of steps to train for. If provided, overrides train_epochs.
        eval_steps=0,                        # Number of steps to eval for. If not provided, evals over entire dataset.
        eval_batch_size=256,                 # Batch size for eval.
        checkpoint_epochs=1,                 # Number of epochs between checkpoints/summaries.
        checkpoint_steps=0,                  # Number of steps between checkpoints/summaries. If provided, overrides checkpoint_epochs.
        eval_split='validation',             # Split for evaluation.
        dataset='imagenet2012',              # Name of a dataset.
        mode='train',                        # Whether to perform training or evaluation. 'train', 'eval', or 'train_then_eval'
        train_mode='pretrain',               # The train mode controls different objectives and trainable components.
        lineareval_while_pretraining=True,   # Whether to finetune supervised head while pretraining. 'pretrain' or 'finetune'
        zero_init_logits_layer=False,        # If True, zero initialize layers after avg_pool for supervised learning.
        fine_tune_after_block=-1,            # The layers after which block that we will fine-tune. -1 means fine-tuning everything. 0 means fine-tuning after stem block. 4 means fine-tuning just the linear head.
        master=None,                         # Address/name of the TensorFlow master to use. By default, use an in-process master.
        data_dir=None,                       # Directory where dataset is stored.
        optimizer='lars',                    # Optimizer to use. 'momentum', 'adam', 'lars'
        momentum=0.9,                        # Momentum parameter.
        keep_checkpoint_max=5,               # Maximum number of checkpoints to keep.
        keep_hub_module_max=1,               # Maximum number of Hub modules to keep.
        temperature=0.1,                     # Temperature parameter for contrastive loss.
        hidden_norm=True,                    # Temperature parameter for contrastive loss.
        proj_head_mode='nonlinear',          # How the head projection is done. 'none', 'linear', 'nonlinear'
        proj_out_dim=128,                    # Number of head projection dimension.
        num_proj_layers=3,                   # Number of non-linear head layers.
        ft_proj_selector=0,                  # Which layer of the projection head to use during fine-tuning. 0 means no projection head, and -1 means the final layer.
        global_bn=True,                      # Whether to aggregate BN statistics across distributed cores.
        width_multiplier=1,                  # Multiplier to change width of network.
        resnet_depth=50,                     # Depth of ResNet.
        sk_ratio=0.,                         # If it is bigger than 0, it will enable SK. Recommendation: 0.0625.
        se_ratio=0.,                         # If it is bigger than 0, it will enable SE.
        image_size=224,                      # Input image size.
        color_jitter_strength=1.0,           # The strength of color jittering.
        use_blur=True,                       # Whether or not to use Gaussian blur for augmentation during pretraining.
        num_classes=None,                    # Number of classes for the supervised head.
    ) -> None:
        """SimCLR arguments."""
        for argname, argval in dict(locals()).items():
            setattr(self, argname, argval)
        #TODO: __dict__
