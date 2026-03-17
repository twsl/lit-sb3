from typing import SupportsFloat

import gymnasium as gym
import numpy as np


class ClipRewardEnv(gym.RewardWrapper):
    """Clip the reward to {+1, 0, -1} by its sign.

    Args:
        env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        """Bin reward to {+1, 0, -1} by its sign.

        Args:
            reward:
            reward: The reward to process.
        """
        return np.sign(float(reward))
