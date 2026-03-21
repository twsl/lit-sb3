from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
import numpy as np


class ActionNoiseWrapper(gym.Wrapper[ObsType, np.ndarray, ObsType, np.ndarray]):
    """Add Gaussian noise to the action (without telling the agent).

    Useful to test the robustness of the control.

    :param env: Gym environment.
    :param noise_std: Standard deviation of the Gaussian noise.
    """

    def __init__(self, env: gym.Env, noise_std: float = 0.1) -> None:
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action: np.ndarray) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert isinstance(self.action_space, spaces.Box)
        noise = np.random.normal(np.zeros_like(action), np.ones_like(action) * self.noise_std)
        noisy_action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return self.env.step(noisy_action)
