"""Common aliases for type hints."""

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Protocol, SupportsFloat

import gymnasium as gym
import numpy as np
import torch

# Avoid circular imports, we use type hint as string to avoid it too
if TYPE_CHECKING:
    from lit_sb3.callbacks import BaseCallback
    from lit_sb3.vec_env import VecEnv

type GymEnv = gym.Env | VecEnv
type GymObs = tuple | dict[str, Any] | np.ndarray | int
type GymResetReturn = tuple[GymObs, dict]
type AtariResetReturn = tuple[np.ndarray, dict[str, Any]]
type GymStepReturn = tuple[GymObs, float, bool, bool, dict]
type AtariStepReturn = tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]
type TensorDict = dict[str, torch.Tensor]
type OptimizerStateDict = dict[str, Any]
type MaybeCallback = None | Callable | list[BaseCallback] | BaseCallback
type PyTorchObs = torch.Tensor | TensorDict

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class RNNStates(NamedTuple):
    pi: tuple[torch.Tensor, ...]
    vf: tuple[torch.Tensor, ...]


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    # For n-step replay buffer
    discounts: torch.Tensor | None = None


class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor | None = None


class RolloutReturn(NamedTuple):
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"


class AgentProtocol(Protocol):
    """Structural protocol for RL agents.

    Used for ``agent: AgentProtocol | None`` type annotations on algorithm
    classes before the full ``BaseAgent`` ABC is wired in (Phase 2).
    Import the concrete ABC from ``lit_sb3.agents`` for runtime checks.
    """

    def configure_optimizers(self) -> list[Any]: ...

    def compute_loss(self, batch: Any) -> dict[str, Any]: ...

    def get_action(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]: ...

    def set_training_mode(self, mode: bool) -> None: ...


class PolicyPredictor(Protocol):
    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Get the policy action from an observation (and optional hidden state).

        Includes sugar-coating to handle different observations (e.g. normalizing images).

        Args:
            observation: the input observation
            state: The last hidden states (can be None, used in recurrent policies)
            episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            deterministic: Whether or not to return deterministic actions.

        Returns:
            the model's action and the next hidden state
            (used in recurrent policies)
        """
