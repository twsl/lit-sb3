from collections.abc import Generator
from typing import NamedTuple

from gymnasium import spaces
import numpy as np
import torch

from lit_sb3.buffers.dict_rollout import DictRolloutBuffer
from lit_sb3.core.types import TensorDict
from lit_sb3.vec_env import VecNormalize


class MaskableDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor


class MaskableDictRolloutBuffer(DictRolloutBuffer):
    """Dict Rollout buffer used in on-policy algorithms like A2C/PPO.

    Extends the RolloutBuffer to use dictionary observations.

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    Args:
        buffer_size: Max number of element in the buffer
        observation_space: Observation space
        action_space: Action space
        device: PyTorch device
        gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Equivalent to classic advantage when set to 1.
        gamma: Discount factor
        n_envs: Number of parallel environments
    """

    action_masks: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ) -> None:
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = int(self.action_space.n)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise TypeError(f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

        super().reset()

    def add(self, *args, action_masks: np.ndarray | None = None, **kwargs) -> None:
        """Add transition to buffer.

        Args:
            *args: Positional arguments forwarded to parent ``add``.
            action_masks: Masks applied to constrain the choice of possible actions.
            **kwargs: Keyword arguments forwarded to parent ``add``.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def get(self, batch_size: int | None = None) -> Generator[MaskableDictRolloutBufferSamples, None, None]:  # type: ignore[override]
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns", "action_masks"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> MaskableDictRolloutBufferSamples:  # type: ignore[override]
        return MaskableDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )
