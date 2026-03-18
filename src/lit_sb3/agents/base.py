"""BaseAgent — abstract base class for RL agents.

An *agent* owns the policy (neural network) and the learning objective:
- ``compute_loss(batch)`` — pure loss computation, returns a dict of tensors.
- ``configure_optimizers()`` — returns the list of optimizers for the policy.
- ``get_action(obs)``  — action selection (overridable for ε-greedy, noise, …).

The *algorithm* (``BaseAlgorithm``) remains the ``LightningModule`` orchestrator
that owns the environment, replay/rollout buffer, data collection loop, and
the Lightning hooks (``training_step``, ``log_dict``, ``manual_backward``).
Agents never call Lightning methods directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from lightning.pytorch.utilities.types import OptimizerLRScheduler
import numpy as np
import torch
from torch.optim import Optimizer

if TYPE_CHECKING:
    from lit_sb3.policies import BasePolicy


class BaseAgent(ABC):
    """Abstract base for RL agents.

    Subclasses must implement :meth:`configure_optimizers` and
    :meth:`compute_loss`.  :meth:`get_action` and :meth:`set_training_mode`
    have sensible defaults that delegate to ``self.policy``.

    ``policy`` must be set by the subclass ``__init__`` (or ``_setup``) before
    any method is called.
    """

    policy: BasePolicy

    def __init__(self, policy: BasePolicy) -> None:
        self.policy = policy

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every concrete agent
    # ------------------------------------------------------------------

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the list of optimizers used by this agent.

        Called once by the algorithm's ``configure_optimizers`` Lightning hook.
        """

    @abstractmethod
    def compute_loss(self, batch: Any) -> dict[str, torch.Tensor]:
        """Compute and return losses for *batch*.

        Returns a ``dict`` mapping loss names to scalar tensors.  The dict must
        contain at least a ``"loss"`` key (the total loss to differentiate).
        All other entries are treated as diagnostic scalars for logging.

        The *algorithm* is responsible for calling ``manual_backward`` and
        ``log_dict`` — this method must not do so itself.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — override as needed
    # ------------------------------------------------------------------

    def get_action(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Select an action given *observation*.

        Default implementation delegates to ``self.policy.predict``.
        Override for exploration strategies (ε-greedy, action noise, …).
        """
        return self.policy.predict(observation, state, episode_start, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """Toggle train/eval mode on the underlying policy."""
        self.policy.set_training_mode(mode)
