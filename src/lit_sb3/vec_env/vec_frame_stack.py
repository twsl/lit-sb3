from collections.abc import Mapping
from typing import Any, Optional, Union

from gymnasium import spaces
import numpy as np

from lit_sb3.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from lit_sb3.vec_env.stacked_observations import StackedObservations


class VecFrameStack(VecEnvWrapper):
    """Frame stacking wrapper for vectorized environment. Designed for image observations.

    Args:
        venv: Vectorized environment to wrap
        n_stack: Number of frames to stack
        channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
            If None, automatically detect channel to stack over in case of image observation or default to "last"
            (default).
            Alternatively channels_order can be a dictionary which can be used with environments with Dict observation
            spaces
    """

    def __init__(self, venv: VecEnv, n_stack: int, channels_order: str | Mapping[str, str] | None = None) -> None:
        assert isinstance(venv.observation_space, (spaces.Box, spaces.Dict)), (
            "VecFrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces"
        )

        self.stacked_obs = StackedObservations(venv.num_envs, n_stack, venv.observation_space, channels_order)
        observation_space = self.stacked_obs.stacked_observation_space
        super().__init__(venv, observation_space=observation_space)

    def step_wait(
        self,
    ) -> tuple[
        np.ndarray | dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        observations, rewards, dones, infos = self.venv.step_wait()
        observations, infos = self.stacked_obs.update(observations, dones, infos)
        return observations, rewards, dones, infos

    def reset(self) -> np.ndarray | dict[str, np.ndarray]:
        """Reset all environments."""
        observation = self.venv.reset()
        observation = self.stacked_obs.reset(observation)
        return observation
