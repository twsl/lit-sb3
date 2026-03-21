from __future__ import annotations

import gymnasium as gym
import numpy as np

from lit_sb3.core.types import GymResetReturn, GymStepReturn


class ActionSmoothingWrapper(gym.Wrapper):
    """Smooth the action using exponential moving average.

    :param env: Gym environment.
    :param smoothing_coef: Smoothing coefficient (0 = no smoothing, 1 = very smooth).
    """

    def __init__(self, env: gym.Env, smoothing_coef: float = 0.0) -> None:
        super().__init__(env)
        self.smoothing_coef = smoothing_coef
        self.smoothed_action: np.ndarray | None = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> GymResetReturn:
        self.smoothed_action = None
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> GymStepReturn:
        if self.smoothed_action is None:
            self.smoothed_action = np.zeros_like(action)
        assert self.smoothed_action is not None
        self.smoothed_action = self.smoothing_coef * self.smoothed_action + (1 - self.smoothing_coef) * action
        return self.env.step(self.smoothed_action)  # ty:ignore[invalid-return-type]
