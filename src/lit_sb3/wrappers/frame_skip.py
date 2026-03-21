from __future__ import annotations

from typing import Any

import gymnasium as gym

from lit_sb3.core.types import GymStepReturn


class FrameSkip(gym.Wrapper):
    """Return only every ``skip``-th frame (frameskipping).

    :param env: Gym environment.
    :param skip: Number of frames to skip (repeat action for).
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: Any) -> GymStepReturn:
        """Step the environment, repeating action and summing reward.

        :param action: The action to repeat.
        :return: observation, cumulative reward, terminated, truncated, info.
        """
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
