from typing import NamedTuple

import torch

from lit_sb3.common.types import TensorDict


class RNNStates(NamedTuple):
    pi: tuple[torch.Tensor, ...]
    vf: tuple[torch.Tensor, ...]
