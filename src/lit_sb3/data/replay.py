"""PyTorch IterableDataset wrappers that bridge RL data collection.

with Lightning's DataLoader-based training loop.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from itertools import count
from typing import TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from lit_sb3.buffers import ReplayBuffer, RolloutBuffer
    from lit_sb3.core.types import RolloutBufferSamples


class ReplayBufferDataset(IterableDataset):
    """Finite IterableDataset for off-policy algorithms.

    Yields a fixed number of dummy integer tokens. The actual data is NOT
    provided by this dataset - instead, ``training_step`` handles replay buffer
    sampling. This dataset simply drives the Lightning training loop for a
    specified number of steps.

    Args:
        total_steps: Total number of training steps (= timer for entire learn()).
    """

    def __init__(self, total_steps: int) -> None:
        super().__init__()
        self.total_steps = total_steps

    def __iter__(self) -> Iterator[int]:
        yield from range(self.total_steps)

    def __len__(self) -> int:
        return self.total_steps


class InfiniteReplayBufferDataset(IterableDataset):
    """Unbounded IterableDataset used when trainer-side RL budget control is authoritative.

    Yields dummy integer tokens forever. The actual data is NOT provided by this
    dataset - instead, ``training_step`` handles replay buffer sampling. This
    dataset exists purely to keep Lightning advancing batches until the trainer
    or callback-driven RL budget requests a stop.
    """

    def __iter__(self) -> Iterator[int]:
        yield from count()
