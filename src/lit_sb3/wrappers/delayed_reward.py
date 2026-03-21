from __future__ import annotations

from typing import Any

import gymnasium as gym

from lit_sb3.core.types import GymResetReturn, GymStepReturn


class DelayedRewardWrapper(gym.Wrapper):
    """Delay the reward by ``delay`` steps (accumulated until released).

    Makes the task harder but more realistic.

    :param env: Gym environment.
    :param delay: Number of steps to delay the reward.
    """

    def __init__(self, env: gym.Env, delay: int = 10) -> None:
        super().__init__(env)
        self.delay = delay
        self.current_step = 0
        self.accumulated_reward = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None) -> GymResetReturn:
        self.current_step = 0
        self.accumulated_reward = 0.0
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.accumulated_reward += float(reward)
        self.current_step += 1

        if self.current_step % self.delay == 0 or terminated or truncated:
            reward = self.accumulated_reward
            self.accumulated_reward = 0.0
        else:
            reward = 0.0
        return obs, reward, terminated, truncated, info
