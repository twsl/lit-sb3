from abc import ABC, abstractmethod

from gymnasium import spaces
import numpy as np
import torch

from lit_sb3.core.preprocessing import get_action_dim, get_obs_shape
from lit_sb3.core.types import ReplayBufferSamples, RolloutBufferSamples
from lit_sb3.utils.device import get_device
from lit_sb3.vec_env import VecNormalize


class BaseBuffer(ABC):
    """Base class that represent a buffer (rollout or replay).

    Args:
        buffer_size: Max number of element in the buffer
        observation_space: Observation space
        action_space: Action space
        device: PyTorch device
            to which the values will be converted
        n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        n_envs: int = 1,
    ) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """Swap and then flatten axes 0 (buffer_size) and 1 (n_envs).

        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order).

        Args:
            arr:
            arr: The array to process.
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """Returns.

        The current size of the buffer.
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """Add elements to the buffer."""
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """Add a new batch of transitions to the buffer."""
        # Do a for loop along the batch axis
        for data in zip(*args, strict=False):
            self.add(*data)

    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> ReplayBufferSamples | RolloutBufferSamples:
        """Args.

        batch_size: Number of element to sample
        env: associated gym VecEnv
            to normalize the observations/rewards when sampling.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: VecNormalize | None = None
    ) -> ReplayBufferSamples | RolloutBufferSamples:
        """Get samples from the buffer.

        Args:
            batch_inds: Indices of samples to get.
            env: Associated VecEnv for normalization (optional).
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """Convert a numpy array to a PyTorch tensor.

        Note: it copies the data by default.

        Args:
            array:
            copy: Whether to copy or not the data (may be useful to avoid changing things
                by reference). This argument is inoperative if the device is not the CPU.
            array: The array to convert to a tensor.
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: np.ndarray | dict[str, np.ndarray],
        env: VecNormalize | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: VecNormalize | None = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward
