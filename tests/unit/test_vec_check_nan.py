"""Unit tests for VecCheckNan wrapper.

Adapted from HenryJia/lightning-baselines3 tests/common/vec_env/test_vec_check_nan.py.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from lit_sb3.vec_env import DummyVecEnv, VecCheckNan


class NanAndInfEnv(gym.Env):
    """Custom environment that returns NaN and Inf observations for testing."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    def step(self, action):
        if np.all(np.array(action) > 0):
            obs = float("NaN")
        elif np.all(np.array(action) < 0):
            obs = float("inf")
        else:
            obs = 0.0
        return [obs], 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return [0.0], {}

    def render(self):
        pass


def test_check_nan():
    """Test VecCheckNan correctly detects NaN and Inf observations."""
    env = DummyVecEnv([NanAndInfEnv])
    env = VecCheckNan(env, raise_exception=True)

    env.step([[0]])

    with pytest.raises(ValueError):
        env.step([[float("NaN")]])

    with pytest.raises(ValueError):
        env.step([[float("inf")]])

    with pytest.raises(ValueError):
        env.step([[-1]])

    with pytest.raises(ValueError):
        env.step([[1]])

    env.reset()
