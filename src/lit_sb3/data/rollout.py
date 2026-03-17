"""PyTorch IterableDataset wrappers that bridge RL data collection.

with Lightning's DataLoader-based training loop.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from lit_sb3.buffers import ReplayBuffer, RolloutBuffer
    from lit_sb3.common.types import RolloutBufferSamples


class RolloutDataset(IterableDataset):
    """IterableDataset wrapping a RolloutBuffer for on-policy algorithms.

    The dataset yields mini-batches from the rollout buffer.
    It is intended to be iterated once per RL training iteration:
    - on_train_epoch_start() collects a fresh rollout and fills the buffer
    - Each __iter__ call yields n_epochs * ceil(buffer_size / batch_size) batches
      (i.e., n_epochs full passes over the buffer data)

    Because this holds a *reference* to the buffer (not a copy), it automatically
    serves freshly collected data each epoch without needing to be recreated.

    Args:
        rollout_buffer: RolloutBuffer to iterate over.
        batch_size: Mini-batch size for each gradient step.
        n_epochs: Number of passes over the rollout buffer per RL iteration.
    """

    def __init__(self, rollout_buffer: RolloutBuffer, batch_size: int, n_epochs: int) -> None:
        super().__init__()
        self.rollout_buffer = rollout_buffer
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def __iter__(self) -> Generator[RolloutBufferSamples, None, None]:
        for _ in range(self.n_epochs):
            yield from self.rollout_buffer.get(self.batch_size)
