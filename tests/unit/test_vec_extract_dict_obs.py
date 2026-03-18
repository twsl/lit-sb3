"""Unit tests for VecExtractDictObs wrapper.

Adapted from stable-baselines3 tests/test_vec_extract_dict_obs.py.
"""

from gymnasium import spaces
import numpy as np
import pytest

from lit_sb3.vec_env.base_vec_env import VecEnv
from lit_sb3.vec_env.vec_extract_dict_obs import VecExtractDictObs
from lit_sb3.vec_env.vec_monitor import VecMonitor


class DictObsVecEnv(VecEnv):
    """Custom VecEnv that produces dictionary observations."""

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.num_envs = 4
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({"rgb": spaces.Box(low=0.0, high=255.0, shape=(86, 86), dtype=np.float32)})
        self.n_steps = 0
        self.max_steps = 5
        self.render_mode = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.n_steps += 1
        done = self.n_steps >= self.max_steps
        if done:
            infos = [
                {
                    "terminal_observation": {"rgb": np.zeros((86, 86), dtype=np.float32)},
                    "TimeLimit.truncated": True,
                }
                for _ in range(self.num_envs)
            ]
        else:
            infos = []
        return (
            {"rgb": np.zeros((self.num_envs, 86, 86), dtype=np.float32)},
            np.zeros((self.num_envs,), dtype=np.float32),
            np.ones((self.num_envs,), dtype=bool) * done,
            infos,
        )

    def reset(self):
        self.n_steps = 0
        return {"rgb": np.zeros((self.num_envs, 86, 86), dtype=np.float32)}

    def render(self, mode=""):
        pass

    def get_attr(self, attr_name, indices=None):
        indices = range(self.num_envs) if indices is None else indices
        return [getattr(self, attr_name) for _ in indices]

    def close(self):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        indices = range(self.num_envs) if indices is None else indices
        return [False for _ in indices]

    def env_method(self):  # ty:ignore[invalid-method-override]
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError


def test_extract_dict_obs():
    """Test VecExtractDictObs."""
    env = DictObsVecEnv()
    env = VecExtractDictObs(env, "rgb")
    assert env.reset().shape == (4, 86, 86)

    for _ in range(10):
        obs, _, dones, infos = env.step([env.action_space.sample() for _ in range(env.num_envs)])  # ty:ignore[invalid-argument-type]
        assert obs.shape == (4, 86, 86)  # ty:ignore[unresolved-attribute]
        for idx, info in enumerate(infos):
            if "terminal_observation" in info:
                assert dones[idx]
                assert info["terminal_observation"].shape == (86, 86)
            else:
                assert not dones[idx]


def test_extract_dict_obs_invalid_key():
    """Test that VecExtractDictObs raises an error for non-Dict spaces."""
    import gymnasium as gym

    from lit_sb3.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    with pytest.raises(AssertionError):
        VecExtractDictObs(env, "some_key")
