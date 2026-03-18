"""OnPolicyAgent — base for advantage-based agents (PPO, A2C, TRPO, …).

On-policy agents receive *mini-batches* from a pre-collected rollout buffer.
One call to :meth:`compute_loss` processes a single mini-batch.

``configure_optimizers`` is made concrete here: every on-policy agent uses
exactly ``[self.policy.optimizer]``.  PPO and A2C are identical in this regard.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch

from lit_sb3.agents.base import BaseAgent


class OnPolicyAgent(BaseAgent):
    """Abstract base for on-policy agents.

    Provides a concrete :meth:`configure_optimizers` (single policy optimizer).
    Subclasses must implement :meth:`compute_loss`.
    """

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the single policy optimizer."""
        return [self.policy.optimizer]

    @abstractmethod
    def compute_loss(self, batch: Any) -> dict[str, torch.Tensor]:
        """Compute on-policy loss for one rollout mini-batch."""
