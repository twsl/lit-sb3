"""Isolation tests for on-policy _compute_loss methods.

These verify that PPO._compute_loss and A2C._compute_loss work correctly
without requiring a real gymnasium environment — confirming the seam is clean
and the methods are safe to migrate to an Agent class in Phase 3.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from lit_sb3.algorithms.ppo import PPO


def _make_rollout_batch(
    obs_dim: int = 4,
    action_dim: int = 1,
    batch_size: int = 8,
    discrete: bool = False,
) -> MagicMock:
    """Return a mock RolloutBufferSamples-like batch."""
    batch = MagicMock()
    batch.observations = torch.randn(batch_size, obs_dim)
    if discrete:
        batch.actions = torch.randint(0, 2, (batch_size,)).float()
    else:
        batch.actions = torch.randn(batch_size, action_dim)
    batch.old_log_prob = torch.randn(batch_size)
    batch.old_values = torch.randn(batch_size)
    batch.returns = torch.randn(batch_size)
    batch.advantages = torch.randn(batch_size)
    return batch


def _make_ppo(obs_dim: int = 4, n_actions: int = 2) -> PPO:  # noqa: F821

    with patch("gymnasium.make") as mock_make:
        mock_env = MagicMock()
        mock_env.observation_space.shape = (obs_dim,)
        mock_env.action_space.n = n_actions
        mock_env.action_space.shape = ()
        mock_env.spec = None
        mock_make.return_value = mock_env
        model = PPO("MlpPolicy", mock_env, n_steps=16, batch_size=8, n_epochs=1)
    return model


def _make_a2c(obs_dim: int = 4, n_actions: int = 2) -> A2C:  # noqa: F821
    from lit_sb3.algorithms.a2c import A2C

    with patch("gymnasium.make") as mock_make:
        mock_env = MagicMock()
        mock_env.observation_space.shape = (obs_dim,)
        mock_env.action_space.n = n_actions
        mock_env.action_space.shape = ()
        mock_env.spec = None
        mock_make.return_value = mock_env
        model = A2C("MlpPolicy", mock_env, n_steps=16)
    return model


class TestPPOComputeLoss:
    """PPO._compute_loss is isolated from the Lightning training loop."""

    def test_returns_required_keys(self, simple_ppo):
        batch = _make_rollout_batch()
        losses = simple_ppo._compute_loss(batch)
        assert set(losses) >= {"loss", "policy_loss", "value_loss", "entropy_loss", "approx_kl", "clip_range"}

    def test_all_values_are_tensors(self, simple_ppo):
        batch = _make_rollout_batch()
        losses = simple_ppo._compute_loss(batch)
        for key, val in losses.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a Tensor"

    def test_loss_is_scalar(self, simple_ppo):
        batch = _make_rollout_batch()
        losses = simple_ppo._compute_loss(batch)
        assert losses["loss"].shape == torch.Size([])

    def test_no_env_required(self, simple_ppo):
        """_compute_loss must not access self.env."""
        simple_ppo.env = None
        batch = _make_rollout_batch()
        # Should not raise — env is not touched
        losses = simple_ppo._compute_loss(batch)
        assert "loss" in losses

    def test_backprop_works(self, simple_ppo):
        batch = _make_rollout_batch()
        losses = simple_ppo._compute_loss(batch)
        losses["loss"].backward()  # must not raise


class TestA2CComputeLoss:
    """A2C._compute_loss is isolated from the Lightning training loop."""

    def test_returns_required_keys(self, simple_a2c):
        batch = _make_rollout_batch()
        losses = simple_a2c._compute_loss(batch)
        assert set(losses) >= {"loss", "policy_loss", "value_loss", "entropy_loss"}

    def test_all_values_are_tensors(self, simple_a2c):
        batch = _make_rollout_batch()
        losses = simple_a2c._compute_loss(batch)
        for key, val in losses.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a Tensor"

    def test_loss_is_scalar(self, simple_a2c):
        batch = _make_rollout_batch()
        losses = simple_a2c._compute_loss(batch)
        assert losses["loss"].shape == torch.Size([])

    def test_no_env_required(self, simple_a2c):
        simple_a2c.env = None
        batch = _make_rollout_batch()
        losses = simple_a2c._compute_loss(batch)
        assert "loss" in losses

    def test_backprop_works(self, simple_a2c):
        batch = _make_rollout_batch()
        losses = simple_a2c._compute_loss(batch)
        losses["loss"].backward()


@pytest.fixture(scope="module")
def simple_ppo():
    """Minimal PPO instance with a real policy but no real env."""
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    env = MagicMock()
    env.observation_space = obs_space
    env.action_space = act_space
    env.num_envs = 1
    env.spec = None

    from lit_sb3.algorithms.ppo import PPO

    model = PPO.__new__(PPO)
    # Minimal init to get a policy without gym env setup
    model.observation_space = obs_space
    model.action_space = act_space
    model.n_envs = 1
    model.device = torch.device("cpu")
    model.verbose = 0
    model.normalize_advantage = True
    model.clip_range = lambda _: 0.2
    model.clip_range_vf = None
    model.target_kl = None
    model.ent_coef = 0.0
    model.vf_coef = 0.5
    model._current_progress_remaining = 1.0
    model._n_updates = 0
    model.env = None

    from lit_sb3.core.utils import FloatSchedule
    from lit_sb3.policies import ActorCriticPolicy

    model.policy = ActorCriticPolicy(
        obs_space,
        act_space,
        lr_schedule=FloatSchedule(3e-4),
    )
    return model


@pytest.fixture(scope="module")
def simple_a2c():
    """Minimal A2C instance with a real policy but no real env."""
    from gymnasium import spaces
    import numpy as np

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    from lit_sb3.algorithms.a2c import A2C

    model = A2C.__new__(A2C)
    model.observation_space = obs_space
    model.action_space = act_space
    model.n_envs = 1
    model.device = torch.device("cpu")
    model.verbose = 0
    model.normalize_advantage = False
    model.ent_coef = 0.0
    model.vf_coef = 0.5
    model._current_progress_remaining = 1.0
    model._n_updates = 0
    model.env = None

    from lit_sb3.core.utils import FloatSchedule
    from lit_sb3.policies import ActorCriticPolicy

    model.policy = ActorCriticPolicy(
        obs_space,
        act_space,
        lr_schedule=FloatSchedule(7e-4),
    )
    return model
