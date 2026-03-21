"""Training utilities: seeding, learning rate, rollout control."""

import random

import numpy as np
import torch

from lit_sb3.core.types import TrainFreq, TrainFrequencyUnit


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """Seed the different random generators.

    Args:
        seed: Random seed for reproducibility.
        using_cuda: Whether CUDA is being used.
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Update the learning rate for a given optimizer.

    Useful when doing linear schedule.

    Args:
        optimizer: Pytorch optimizer
        learning_rate: New learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def should_collect_more_steps(
    train_freq: TrainFreq,
    num_collected_steps: int,
    num_collected_episodes: int,
) -> bool:
    """Helper used in ``collect_rollouts()`` of off-policy algorithms.

    to determine the termination condition.

    Args:
        train_freq: How much experience should be collected before updating the policy.
        num_collected_steps: The number of already collected steps.
        num_collected_episodes: The number of already collected episodes.

    Returns:
        Whether to continue or not collecting experience
        by doing rollouts of the current policy.
    """
    if train_freq.unit == TrainFrequencyUnit.STEP:
        return num_collected_steps < train_freq.frequency

    elif train_freq.unit == TrainFrequencyUnit.EPISODE:
        return num_collected_episodes < train_freq.frequency

    else:
        raise ValueError(
            "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
            f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
        )
