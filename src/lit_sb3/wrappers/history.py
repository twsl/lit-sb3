from __future__ import annotations

from typing import SupportsFloat

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HistoryWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Stack past observations and actions to give a history to the agent.

    :param env: Gym environment.
    :param horizon: Number of past steps to include in the observation.
    """

    def __init__(self, env: gym.Env, horizon: int = 2) -> None:
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        low_obs = np.tile(wrapped_obs_space.low, horizon)
        high_obs = np.tile(wrapped_obs_space.high, horizon)

        low_action = np.tile(wrapped_action_space.low, horizon)
        high_action = np.tile(wrapped_action_space.high, horizon)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space = spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)  # type: ignore[arg-type]

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        assert options is None, "Options not supported for now"
        obs, info = self.env.reset(seed=seed)
        self.obs_history[..., -obs.shape[-1] :] = obs
        return self._create_obs_from_history(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action
        return self._create_obs_from_history(), reward, terminated, truncated, info


class HistoryWrapperObsDict(gym.Wrapper):
    """History Wrapper for dict observation spaces.

    :param env: Gym environment with a dict observation space.
    :param horizon: Number of past steps to include.
    """

    def __init__(self, env: gym.Env, horizon: int = 2) -> None:
        assert isinstance(env.observation_space, spaces.Dict)
        assert isinstance(env.observation_space.spaces["observation"], spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

        wrapped_obs_space = env.observation_space.spaces["observation"]
        wrapped_action_space = env.action_space

        low_obs = np.tile(wrapped_obs_space.low, horizon)
        high_obs = np.tile(wrapped_obs_space.high, horizon)

        low_action = np.tile(wrapped_action_space.low, horizon)
        high_action = np.tile(wrapped_action_space.high, horizon)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space.spaces["observation"] = spaces.Box(
            low=low,
            high=high,
            dtype=wrapped_obs_space.dtype,  # type: ignore[arg-type]
        )

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        assert options is None, "Options not supported for now"
        obs_dict, info = self.env.reset(seed=seed)
        obs = obs_dict["observation"]
        self.obs_history[..., -obs.shape[-1] :] = obs

        obs_dict["observation"] = self._create_obs_from_history()

        return obs_dict, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict]:
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        obs = obs_dict["observation"]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action

        obs_dict["observation"] = self._create_obs_from_history()

        return obs_dict, reward, terminated, truncated, info
