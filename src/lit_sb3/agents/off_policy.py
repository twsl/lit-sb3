"""OffPolicyAgent — base for replay-buffer-based agents (SAC, TD3, DQN, …).

Off-policy agents learn from transitions stored in a replay buffer.
``configure_optimizers`` is kept abstract because the optimizer set varies
significantly across algorithms (SAC has 3, TD3 has 2, DQN has 1).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch

from lit_sb3.agents.base import BaseAgent


class OffPolicyAgent(BaseAgent):
    """Abstract base for off-policy agents.

    Subclasses must implement both :meth:`configure_optimizers` and
    :meth:`compute_loss`.
    """

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the list of optimizers (varies: SAC=3, TD3=2, DQN=1)."""

    @abstractmethod
    def compute_loss(self, batch: Any) -> dict[str, torch.Tensor]:
        """Compute off-policy loss for one replay-buffer batch."""
