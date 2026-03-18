"""Unit tests for environment checker (check_env).

Adapted from HenryJia/lightning-baselines3 tests/common/test_envs.py.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from lit_sb3.core.env_checker import check_env
from lit_sb3.envs.bit_flipping_env import BitFlippingEnv
from lit_sb3.envs.identity_env import (
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
)

ENV_CLASSES = [
    BitFlippingEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
    FakeImageEnv,
]


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_env(env_id):
    """Check that standard Gym environments pass check_env without warnings.

    :param env_id: Gym environment ID
    """
    import warnings

    env = gym.make(env_id)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        check_env(env)
    user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
    # Pendulum-v1 produces a warning because the action space [-2, 2] is not [-1, 1]
    if env_id == "Pendulum-v1":
        assert len(user_warnings) >= 1
    else:
        assert len(user_warnings) == 0


@pytest.mark.parametrize("env_class", ENV_CLASSES)
def test_custom_envs(env_class):
    env = env_class()
    check_env(env)


def test_high_dimension_action_space():
    """Test check_env with a continuous action space with more than one action."""
    env = FakeImageEnv()
    env.action_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)

    def patched_step(_action):
        return env.observation_space.sample(), 0.0, False, False, {}

    env.step = patched_step  # ty:ignore[invalid-assignment]
    check_env(env)


@pytest.mark.parametrize(
    "new_obs_space",
    [
        # Small image
        spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
        # Range not in [0, 255]
        spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.uint8),
        # Wrong dtype
        spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.float32),
        # Not an image, it should be a 1D vector
        spaces.Box(low=-1, high=1, shape=(64, 3), dtype=np.float32),
        # Tuple space is not supported by SB3
        spaces.Tuple([spaces.Discrete(5), spaces.Discrete(10)]),
    ],
)
def test_non_default_spaces(new_obs_space):
    env = FakeImageEnv()
    env.observation_space = new_obs_space

    def patched_reset(*, seed=None, options=None):
        return new_obs_space.sample(), {}

    def patched_step(_action):
        return new_obs_space.sample(), 0.0, False, False, {}

    env.reset = patched_reset  # ty:ignore[invalid-assignment]
    env.step = patched_step  # ty:ignore[invalid-assignment]
    with pytest.warns(UserWarning):
        check_env(env)


def _check_reset_assert_error(env, wrong_reset_fn):
    """Helper to check that a wrong reset() return is caught by check_env."""
    env.reset = wrong_reset_fn
    with pytest.raises(AssertionError):
        check_env(env)


def test_common_failures_reset():
    """Test that common failure cases of the reset() method are caught."""
    env = IdentityEnvBox()

    # Return an observation that doesn't match the observation space shape
    _check_reset_assert_error(env, lambda **kw: (np.ones((3,)), {}))
    # The observation is not a numpy array
    _check_reset_assert_error(env, lambda **kw: (1, {}))
    # reset() doesn't return a tuple (obs, info)
    _check_reset_assert_error(env, lambda **kw: env.observation_space.sample())
    # reset() returns tuple with wrong number of elements
    _check_reset_assert_error(env, lambda **kw: (env.observation_space.sample(), {}, False))


def _check_step_assert_error(env, wrong_step_fn):
    """Helper to check that a wrong step() return is caught by check_env."""
    env.step = wrong_step_fn
    with pytest.raises(AssertionError):
        check_env(env)


def test_common_failures_step():
    """Test that common failure cases of the step() method are caught."""
    env = IdentityEnvBox()

    # Wrong shape for the observation
    _check_step_assert_error(env, lambda _: (np.ones((4,)), 1.0, False, False, {}))
    # Obs is not a numpy array
    _check_step_assert_error(env, lambda _: (1, 1.0, False, False, {}))
    # Return a wrong reward (numpy array instead of scalar)
    _check_step_assert_error(env, lambda _: (env.observation_space.sample(), np.ones(1), False, False, {}))
    # Info dict is not returned (only 4-tuple instead of 5-tuple)
    _check_step_assert_error(env, lambda _: (env.observation_space.sample(), 0.0, False, False))
    # terminated is not a boolean
    _check_step_assert_error(env, lambda _: (env.observation_space.sample(), 0.0, 3.0, False, {}))
    _check_step_assert_error(env, lambda _: (env.observation_space.sample(), 0.0, 1, False, {}))
    # truncated is not a boolean
    _check_step_assert_error(env, lambda _: (env.observation_space.sample(), 0.0, False, 1, {}))
