from collections.abc import Generator
from typing import NamedTuple

from gymnasium import spaces
import numpy as np
import torch

from lit_sb3.buffers.rollout import RolloutBuffer
from lit_sb3.vec_env import VecNormalize


class MaskableRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor


class MaskableRolloutBuffer(RolloutBuffer):
    """Rollout buffer that also stores the invalid action masks associated with each observation.

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
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

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
        """Args:
        action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def get(self, batch_size: int | None = None) -> Generator[MaskableRolloutBufferSamples, None, None]:  # type: ignore[override]
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> MaskableRolloutBufferSamples:  # type: ignore[override]
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.action_masks[batch_inds].reshape(-1, self.mask_dims),
        )
        return MaskableRolloutBufferSamples(*map(self.to_torch, data))
