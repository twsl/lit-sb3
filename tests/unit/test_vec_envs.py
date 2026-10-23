"""Unit tests for vectorized environments (DummyVecEnv, SubprocVecEnv, etc.).

Adapted from HenryJia/lightning-baselines3 tests/common/vec_env/test_vec_envs.py.
"""

import collections
import functools
import itertools
import multiprocessing
import os
from pathlib import Path
import shutil

import gymnasium as gym
import numpy as np
import pytest

from lit_sb3.utils.env import make_vec_env
from lit_sb3.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecFrameStack
from lit_sb3.wrappers.monitor import Monitor

N_ENVS = 3
VEC_ENV_CLASSES = [DummyVecEnv, SubprocVecEnv]
VEC_ENV_WRAPPERS = [None, VecFrameStack]


class CustomGymEnv(gym.Env):
    """Custom gymnasium environment for testing purposes."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, space, render_mode=None):
        self.action_space = space
        self.observation_space = space
        self.render_mode = render_mode
        self.current_step = 0
        self.ep_length = 4

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self.current_step = 0
        self._choose_next_state()
        return self.state, {}

    def step(self, action):
        reward = 1
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, False, done, {}

    def _choose_next_state(self):
        self.state = self.observation_space.sample()

    def render(self):
        return np.zeros((4, 4, 3))

    def seed(self, seed=None):
        pass

    @staticmethod
    def custom_method(dim_0=1, dim_1=1):
        """Dummy method to test calling custom methods from a VecEnv."""
        return np.ones((dim_0, dim_1))


@pytest.mark.parametrize("env_id", ["CartPole-v1", lambda: gym.make("CartPole-v1")])
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("vec_env_cls", [None, SubprocVecEnv])
@pytest.mark.parametrize("wrapper_class", [None, gym.wrappers.TimeLimit])
def test_make_vec_env(env_id, n_envs, vec_env_cls, wrapper_class):
    wrapper_kwargs = {"max_episode_steps": 500} if wrapper_class is gym.wrappers.TimeLimit else {}
    env = make_vec_env(
        env_id,
        n_envs,
        vec_env_cls=vec_env_cls,
        wrapper_class=wrapper_class,
        wrapper_kwargs=wrapper_kwargs,
        monitor_dir=None,
        seed=0,
    )
    assert env.num_envs == n_envs

    if vec_env_cls is None:
        assert isinstance(env, DummyVecEnv)
        if wrapper_class is not None:
            assert isinstance(env.envs[0], wrapper_class)
        else:
            assert isinstance(env.envs[0], Monitor)
    else:
        assert isinstance(env, SubprocVecEnv)
    env.close()


def test_vec_env_kwargs():
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, env_kwargs={"goal_velocity": 0.11})
    assert env.get_attr("goal_velocity")[0] == 0.11


def test_vec_env_monitor_kwargs():
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, monitor_kwargs={"allow_early_resets": False})
    assert env.get_attr("allow_early_resets")[0] is False

    env = make_vec_env("MountainCarContinuous-v0", n_envs=1, seed=0, monitor_kwargs={"allow_early_resets": True})
    assert env.get_attr("allow_early_resets")[0] is True


def test_vec_env_wrapper():
    env = make_vec_env("CartPole-v1")
    env2 = VecCheckNan(env)
    assert env.unwrapped == env2.unwrapped
    env2.close()


def test_vec_env_getattr():
    env = make_vec_env("CartPole-v1")
    env.foo = 123  # ty:ignore[unresolved-attribute]
    env2 = VecCheckNan(env)
    env2.bar = 123  # ty:ignore[unresolved-attribute]
    assert env.foo == env2.foo  # ty:ignore[unresolved-attribute]
    assert env2.bar == 123
    env2.close()


def test_custom_vec_env(tmp_path):
    """Stand-alone test for passing a custom VecEnv class to make_vec_env."""
    monitor_dir = tmp_path / "test_make_vec_env/"
    env = make_vec_env(
        "CartPole-v1",
        n_envs=1,
        monitor_dir=monitor_dir,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": None},
    )
    assert env.num_envs == 1
    assert isinstance(env, SubprocVecEnv)
    assert Path(monitor_dir).is_dir()
    env.close()
    shutil.rmtree(monitor_dir)

    # This should fail because DummyVecEnv does not accept any keyword arguments
    with pytest.raises(TypeError):
        make_vec_env("CartPole-v1", n_envs=1, vec_env_kwargs={"dummy": False})


@pytest.mark.parametrize("vec_env_class", VEC_ENV_CLASSES)
@pytest.mark.parametrize("vec_env_wrapper", VEC_ENV_WRAPPERS)
def test_vecenv_custom_calls(vec_env_class, vec_env_wrapper):
    """Test access to methods/attributes of vectorized environments."""

    def make_env():
        return CustomGymEnv(gym.spaces.Box(low=np.zeros(2), high=np.ones(2)), render_mode="rgb_array")

    vec_env = vec_env_class([make_env for _ in range(N_ENVS)])

    if vec_env_wrapper is not None:
        vec_env = vec_env_wrapper(vec_env, n_stack=2) if vec_env_wrapper == VecFrameStack else vec_env_wrapper(vec_env)

    vec_env.seed(0)
    vec_env.render(mode="rgb_array")
    env_method_results = vec_env.env_method("custom_method", 1, indices=None, dim_1=2)
    setattr_results = []
    for env_idx in range(N_ENVS):
        setattr_results.append(vec_env.set_attr("current_step", env_idx, indices=env_idx))
    getattr_results = vec_env.get_attr("current_step")

    assert len(env_method_results) == N_ENVS
    assert len(setattr_results) == N_ENVS
    assert len(getattr_results) == N_ENVS

    for env_idx in range(N_ENVS):
        assert (env_method_results[env_idx] == np.ones((1, 2))).all()
        assert setattr_results[env_idx] is None
        assert getattr_results[env_idx] == env_idx

    # Call env_method on a subset of the VecEnv
    env_method_subset = vec_env.env_method("custom_method", 1, indices=[0, 2], dim_1=3)
    assert (env_method_subset[0] == np.ones((1, 3))).all()
    assert (env_method_subset[1] == np.ones((1, 3))).all()
    assert len(env_method_subset) == 2

    # Test set_attr for all environments
    setattr_result = vec_env.set_attr("current_step", 42, indices=None)
    getattr_result = vec_env.get_attr("current_step")
    assert setattr_result is None
    assert getattr_result == [42 for _ in range(N_ENVS)]

    # Additional tests for set_attr on a subset
    vec_env.reset()
    setattr_result = vec_env.set_attr("current_step", 12, indices=[0, 1])
    getattr_result = vec_env.get_attr("current_step")
    getattr_result_subset = vec_env.get_attr("current_step", indices=[0, 1])
    assert setattr_result is None
    assert getattr_result == [12 for _ in range(2)] + [0 for _ in range(N_ENVS - 2)]
    assert getattr_result_subset == [12, 12]
    assert vec_env.get_attr("current_step", indices=[0, 2]) == [12, 0]

    vec_env.reset()
    setattr_result = vec_env.set_attr("current_step", 12, indices=[0, -1])
    getattr_result = vec_env.get_attr("current_step")
    assert setattr_result is None
    assert getattr_result == [12] + [0 for _ in range(N_ENVS - 2)] + [12]
    assert vec_env.get_attr("current_step", indices=[-1]) == [12]

    vec_env.close()


class StepEnv(gym.Env):
    """Gym environment for testing that terminal_observation is inserted correctly."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(np.array([0]), np.array([999]), dtype="int64")  # ty:ignore[invalid-argument-type]
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        return np.array([self.current_step], dtype="int64"), {}

    def step(self, action):
        prev_step = self.current_step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return np.array([prev_step], dtype="int64"), 0.0, False, done, {}


@pytest.mark.parametrize("vec_env_class", VEC_ENV_CLASSES)
@pytest.mark.parametrize("vec_env_wrapper", VEC_ENV_WRAPPERS)
def test_vecenv_terminal_obs(vec_env_class, vec_env_wrapper):
    """Test that 'terminal_observation' gets added to info dict upon termination."""
    step_nums = [i + 5 for i in range(N_ENVS)]
    vec_env = vec_env_class([functools.partial(StepEnv, n) for n in step_nums])

    if vec_env_wrapper is not None:
        vec_env = vec_env_wrapper(vec_env, n_stack=2) if vec_env_wrapper == VecFrameStack else vec_env_wrapper(vec_env)

    zero_acts = np.zeros((N_ENVS,), dtype="int")
    prev_obs_b = vec_env.reset()
    for step_num in range(1, max(step_nums) + 1):
        obs_b, _, done_b, info_b = vec_env.step(zero_acts)
        assert len(obs_b) == N_ENVS
        assert len(done_b) == N_ENVS
        assert len(info_b) == N_ENVS
        env_iter = zip(prev_obs_b, obs_b, done_b, info_b, step_nums, strict=False)
        for prev_obs, obs, done, info, final_step_num in env_iter:
            assert done == (step_num == final_step_num)
            if not done:
                assert "terminal_observation" not in info
            else:
                terminal_obs = info["terminal_observation"]
                assert np.all(prev_obs < terminal_obs)
                assert np.all(obs < prev_obs)
                assert np.all(prev_obs + 1 == terminal_obs)
                assert np.all(obs == 0)
        prev_obs_b = obs_b

    vec_env.close()


SPACES = collections.OrderedDict(
    [
        ("discrete", gym.spaces.Discrete(2)),
        ("multidiscrete", gym.spaces.MultiDiscrete([2, 3])),
        ("multibinary", gym.spaces.MultiBinary(3)),
        ("continuous", gym.spaces.Box(low=np.zeros(2), high=np.ones(2))),
    ]
)


def check_vecenv_spaces(vec_env_class, space, obs_assert):
    """Helper to check observation spaces in vectorized environments."""

    def make_env():
        return CustomGymEnv(space)

    vec_env = vec_env_class([make_env for _ in range(N_ENVS)])
    obs = vec_env.reset()
    obs_assert(obs)

    dones = [False] * N_ENVS
    while not any(dones):
        actions = [vec_env.action_space.sample() for _ in range(N_ENVS)]
        obs, _rews, dones, _infos = vec_env.step(actions)
        obs_assert(obs)
    vec_env.close()


def check_vecenv_obs(obs, space):
    """Helper to check that each observation from the VecEnv belongs to the observation space."""
    assert obs.shape[0] == N_ENVS
    for value in obs:
        assert space.contains(value)


@pytest.mark.parametrize("vec_env_class,space", itertools.product(VEC_ENV_CLASSES, SPACES.values()))
def test_vecenv_single_space(vec_env_class, space):
    def obs_assert(obs):
        return check_vecenv_obs(obs, space)

    check_vecenv_spaces(vec_env_class, space, obs_assert)


def test_subproc_start_method():
    start_methods = [None]
    all_methods = {"forkserver", "spawn", "fork"}
    available_methods = multiprocessing.get_all_start_methods()
    start_methods += list(all_methods.intersection(available_methods))
    space = gym.spaces.Discrete(2)

    def obs_assert(obs):
        return check_vecenv_obs(obs, space)

    for start_method in start_methods:
        vec_env_class = functools.partial(SubprocVecEnv, start_method=start_method)
        check_vecenv_spaces(vec_env_class, space, obs_assert)

    with pytest.raises(ValueError, match="cannot find context for 'illegal_method'"):
        vec_env_class = functools.partial(SubprocVecEnv, start_method="illegal_method")
        check_vecenv_spaces(vec_env_class, space, obs_assert)


def test_framestack_vecenv():
    """Test that VecFrameStack stacks on the correct axis."""
    image_space_shape = [12, 8, 3]
    zero_acts = np.zeros([N_ENVS] + image_space_shape)

    transposed_image_space_shape = image_space_shape[::-1]
    transposed_zero_acts = np.zeros([N_ENVS] + transposed_image_space_shape)

    def make_image_env():
        return CustomGymEnv(
            gym.spaces.Box(
                low=np.zeros(image_space_shape),
                high=np.ones(image_space_shape) * 255,
                dtype=np.uint8,
            )
        )

    def make_transposed_image_env():
        return CustomGymEnv(
            gym.spaces.Box(
                low=np.zeros(transposed_image_space_shape),
                high=np.ones(transposed_image_space_shape) * 255,
                dtype=np.uint8,
            )
        )

    def make_non_image_env():
        return CustomGymEnv(gym.spaces.Box(low=np.zeros((2,)), high=np.ones((2,))))

    vec_env = DummyVecEnv([make_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2)
    obs, _, _, _ = vec_env.step(zero_acts)
    vec_env.close()
    # Should be stacked on the last dimension
    assert obs.shape[-1] == (image_space_shape[-1] * 2)  # ty:ignore[unresolved-attribute]

    # Try automatic stacking on first dimension (transposed images)
    vec_env = DummyVecEnv([make_transposed_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2)
    obs, _, _, _ = vec_env.step(transposed_zero_acts)
    vec_env.close()
    # Should be stacked on the first dimension
    assert obs.shape[1] == (image_space_shape[-1] * 2)  # ty:ignore[unresolved-attribute]

    # Try forcing channels_order="last"
    vec_env = DummyVecEnv([make_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2, channels_order="last")
    obs, _, _, _ = vec_env.step(zero_acts)
    vec_env.close()
    assert obs.shape[-1] == (image_space_shape[-1] * 2)  # ty:ignore[unresolved-attribute]

    # Try forcing channels_order="first"
    vec_env = DummyVecEnv([make_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2, channels_order="first")
    obs, _, _, _ = vec_env.step(zero_acts)
    vec_env.close()
    assert obs.shape[1] == (image_space_shape[0] * 2)  # ty:ignore[unresolved-attribute]

    # Test invalid channels_order
    vec_env = DummyVecEnv([make_image_env for _ in range(N_ENVS)])
    with pytest.raises(AssertionError):
        vec_env = VecFrameStack(vec_env, n_stack=2, channels_order="not_valid")

    # Test with non-image envs (no channels_order required)
    vec_env = DummyVecEnv([make_non_image_env for _ in range(N_ENVS)])
    vec_env = VecFrameStack(vec_env, n_stack=2)
    obs = vec_env.reset()
    vec_env.close()
