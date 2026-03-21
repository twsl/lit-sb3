from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym

from lit_sb3.core.types import GymResetReturn, GymStepReturn


class TruncatedOnSuccessWrapper(gym.Wrapper):
    """Reset on success and offset the reward.

    Useful for GoalEnv environments where success should end the episode.

    :param env: Gym environment.
    :param reward_offset: Reward added on top of the env reward at each step.
    :param n_successes: Number of consecutive successes required to truncate.
    """

    def __init__(self, env: gym.Env, reward_offset: float = 0.0, n_successes: int = 1) -> None:
        super().__init__(env)
        self.reward_offset = reward_offset
        self.n_successes = n_successes
        self.current_successes = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> GymResetReturn:
        self.current_successes = 0
        assert options is None, "Options not supported for now"
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("is_success", False):
            self.current_successes += 1
        else:
            self.current_successes = 0
        # number of successes in a row
        truncated = truncated or self.current_successes >= self.n_successes
        reward = float(reward) + self.reward_offset
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal: Any, desired_goal: Any, info: dict[str, Any]) -> SupportsFloat:
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)  # ty:ignore[unresolved-attribute]
        return reward + self.reward_offset
