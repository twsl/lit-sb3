"""Unit tests for replay and rollout buffers.

Adapted from stable-baselines3 tests/test_buffers.py.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch as th

from lit_sb3.buffers.dict_replay import DictReplayBuffer
from lit_sb3.buffers.dict_rollout import DictRolloutBuffer
from lit_sb3.buffers.replay import ReplayBuffer
from lit_sb3.buffers.rollout import RolloutBuffer
from lit_sb3.core.env_checker import check_env
from lit_sb3.core.types import DictReplayBufferSamples, ReplayBufferSamples
from lit_sb3.utils.device import get_device
from lit_sb3.utils.env import make_vec_env
from lit_sb3.vec_env import VecNormalize


class DummyEnv(gym.Env):
    """Custom gym environment for testing purposes."""

    def __init__(self):
        self.action_space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Box(1, 5, (1,))
        self._observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = 100

    def reset(self, *, seed=None, options=None):
        self._t = 0
        obs = self._observations[0]
        return obs, {}

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = self._observations[index]
        terminated = False
        truncated = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, terminated, truncated, {}


class DummyDictEnv(gym.Env):
    """Custom gym environment with Dict observations for testing purposes."""

    def __init__(self):
        self.action_space = spaces.Box(1, 5, shape=(10, 7))
        space = spaces.Box(1, 5, (1,))
        self.observation_space = spaces.Dict({"observation": space, "achieved_goal": space, "desired_goal": space})
        self._observations = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        self._rewards = [1, 2, 3, 4, 5]
        self._t = 0
        self._ep_length = 100

    def reset(self, seed=None, options=None):
        self._t = 0
        obs = dict.fromkeys(self.observation_space.spaces.keys(), self._observations[0])  # ty:ignore[unresolved-attribute]
        return obs, {}

    def step(self, action):
        self._t += 1
        index = self._t % len(self._observations)
        obs = dict.fromkeys(self.observation_space.spaces.keys(), self._observations[index])  # ty:ignore[unresolved-attribute]
        terminated = False
        truncated = self._t >= self._ep_length
        reward = self._rewards[index]
        return obs, reward, terminated, truncated, {}


@pytest.mark.parametrize("env_cls", [DummyEnv, DummyDictEnv])
def test_env(env_cls):
    """Check the env used for testing passes the environment checker."""
    check_env(env_cls(), warn=False, skip_render_check=True)


@pytest.mark.parametrize("replay_buffer_cls", [ReplayBuffer, DictReplayBuffer])
def test_replay_buffer_normalization(replay_buffer_cls):
    env_cls = {ReplayBuffer: DummyEnv, DictReplayBuffer: DummyDictEnv}[replay_buffer_cls]
    env = make_vec_env(env_cls)
    env = VecNormalize(env)

    buffer = replay_buffer_cls(100, env.observation_space, env.action_space, device="cpu")

    # Interact and store transitions
    env.reset()
    obs = env.get_original_obs()
    for _ in range(100):
        action = env.action_space.sample()
        _, _, done, info = env.step(action)
        next_obs = env.get_original_obs()
        reward = env.get_original_reward()
        buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

    sample = buffer.sample(50, env)
    # Test observation normalization (mean should be close to ~2 after normalization)
    for observations in [sample.observations, sample.next_observations]:
        if isinstance(sample, DictReplayBufferSamples):
            for key in observations:
                assert th.allclose(observations[key].mean(0), th.zeros(1), atol=1)
        elif isinstance(sample, ReplayBufferSamples):
            assert th.allclose(observations.mean(0), th.zeros(1), atol=1)
    # Test reward normalization
    assert np.allclose(sample.rewards.mean(0), np.zeros(1), atol=1)


@pytest.mark.parametrize(
    "replay_buffer_cls",
    [DictReplayBuffer, DictRolloutBuffer, ReplayBuffer, RolloutBuffer],
)
@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_device_buffer(replay_buffer_cls, device):
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA not available")

    env_cls = {
        RolloutBuffer: DummyEnv,
        DictRolloutBuffer: DummyDictEnv,
        ReplayBuffer: DummyEnv,
        DictReplayBuffer: DummyDictEnv,
    }[replay_buffer_cls]
    env = make_vec_env(env_cls)

    buffer = replay_buffer_cls(100, env.observation_space, env.action_space, device=device)

    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
            episode_start, values, log_prob = np.zeros(1), th.zeros(1), th.ones(1)
            buffer.add(obs, action, reward, episode_start, values, log_prob)
        else:
            buffer.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

    desired_device = get_device(device).type
    if replay_buffer_cls in [RolloutBuffer, DictRolloutBuffer]:
        data = buffer.get(50)
    elif replay_buffer_cls in [ReplayBuffer, DictReplayBuffer]:
        data = [buffer.sample(50)]

    for minibatch in list(data):
        for value in minibatch:
            if isinstance(value, dict):
                for key in value:
                    assert value[key].device.type == desired_device
            elif isinstance(value, th.Tensor):
                assert value.device.type == desired_device


def test_get_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(th.cuda, "is_available", lambda: True)
    monkeypatch.setattr(th.backends.mps, "is_available", lambda: True)

    assert get_device("auto").type == "cuda"


def test_get_device_auto_uses_mps_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(th.cuda, "is_available", lambda: False)
    monkeypatch.setattr(th.backends.mps, "is_available", lambda: True)

    assert get_device("auto").type == "mps"


def test_get_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(th.cuda, "is_available", lambda: False)
    monkeypatch.setattr(th.backends.mps, "is_available", lambda: False)

    assert get_device("auto").type == "cpu"


def test_get_device_unavailable_mps_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(th.backends.mps, "is_available", lambda: False)

    assert get_device("mps").type == "cpu"


@pytest.mark.parametrize(
    "obs_dtype",
    [
        np.dtype(np.uint8),
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.int16),
        np.dtype(np.uint32),
        np.dtype(np.int32),
        np.dtype(np.uint64),
        np.dtype(np.int64),
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    ],
)
@pytest.mark.parametrize("use_dict", [False, True])
@pytest.mark.parametrize(
    "action_space",
    [
        spaces.Discrete(10),
        spaces.Box(low=-1.0, high=1.0, dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, dtype=np.float64),
    ],
)
def test_buffer_dtypes(obs_dtype, use_dict, action_space):
    obs_space = spaces.Box(0, 100, dtype=obs_dtype)
    buffer_params = {"buffer_size": 1, "action_space": action_space}
    actual_replay_action_dtype = ReplayBuffer._maybe_cast_dtype(action_space.dtype)
    # For on-policy, float32 for backward compat
    actual_rollout_action_dtype = np.float32

    if use_dict:
        dict_obs_space = spaces.Dict({"obs": obs_space, "obs_2": spaces.Box(0, 100, dtype=np.uint8)})
        buffer_params["observation_space"] = dict_obs_space
        rollout_buffer = DictRolloutBuffer(**buffer_params)  # ty:ignore[invalid-argument-type]
        replay_buffer = DictReplayBuffer(**buffer_params)  # ty:ignore[invalid-argument-type]
        assert rollout_buffer.observations["obs"].dtype == obs_dtype
        assert replay_buffer.observations["obs"].dtype == obs_dtype
        assert rollout_buffer.observations["obs_2"].dtype == np.uint8
        assert replay_buffer.observations["obs_2"].dtype == np.uint8
    else:
        buffer_params["observation_space"] = obs_space
        rollout_buffer = RolloutBuffer(**buffer_params)  # ty:ignore[invalid-argument-type]
        replay_buffer = ReplayBuffer(**buffer_params)  # ty:ignore[invalid-argument-type]
        assert rollout_buffer.observations.dtype == obs_dtype
        assert replay_buffer.observations.dtype == obs_dtype

    assert rollout_buffer.actions.dtype == action_space.dtype
    assert replay_buffer.actions.dtype == actual_replay_action_dtype

    # Check that sampled types are correct
    rollout_buffer.full = True
    replay_buffer.full = True
    rollout_data = next(rollout_buffer.get(batch_size=64))
    buffer_data = replay_buffer.sample(batch_size=64)
    assert rollout_data.actions.numpy().dtype == actual_rollout_action_dtype
    assert buffer_data.actions.numpy().dtype == actual_replay_action_dtype
    if use_dict:
        assert buffer_data.observations["obs"].numpy().dtype == obs_dtype  # ty:ignore[invalid-argument-type]
        assert buffer_data.observations["obs_2"].numpy().dtype == np.uint8  # ty:ignore[invalid-argument-type]
        assert rollout_data.observations["obs"].numpy().dtype == obs_dtype  # ty:ignore[invalid-argument-type]
        assert rollout_data.observations["obs_2"].numpy().dtype == np.uint8  # ty:ignore[invalid-argument-type]
    else:
        assert buffer_data.observations.numpy().dtype == obs_dtype
        assert rollout_data.observations.numpy().dtype == obs_dtype
