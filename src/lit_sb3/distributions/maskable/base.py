"""Base class for maskable distributions."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from lit_sb3.distributions.base import Distribution

MaybeMasks = torch.Tensor | np.ndarray | None


class MaskableDistribution(Distribution, ABC):
    @abstractmethod
    def apply_masking(self, masks: MaybeMasks) -> None:
        """Eliminate ("mask out") chosen distribution outcomes by setting their probability to 0.

        Args:
            masks: An optional boolean ndarray of compatible shape with the distribution.
                If True, the corresponding choice's logit value is preserved. If False, it is set
                to a large negative value, resulting in near 0 probability. If masks is None, any
                previously applied masking is removed, and the original logits are restored.
        """

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> nn.Module:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes.
        """
