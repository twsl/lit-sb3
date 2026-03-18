"""Integration tests for off-policy algorithms (SAC, TD3, DQN)."""

import pytest


def test_dqn_cartpole():
    """DQN should run on CartPole-v1 for 1000 steps."""
    from lit_sb3.algorithms.dqn.dqn import DQN

    model = DQN("MlpPolicy", "CartPole-v1", learning_starts=100, verbose=0)
    model.learn(total_timesteps=1000)
    assert model.num_timesteps >= 1000


def test_sac_pendulum():
    """SAC should run on Pendulum-v1 for 1000 steps."""
    from lit_sb3.algorithms.sac.sac import SAC

    model = SAC("MlpPolicy", "Pendulum-v1", learning_starts=100, verbose=0)
    model.learn(total_timesteps=1000)
    assert model.num_timesteps >= 1000


def test_td3_pendulum():
    """TD3 should run on Pendulum-v1 for 1000 steps."""
    from lit_sb3.algorithms.td3.td3 import TD3

    model = TD3("MlpPolicy", "Pendulum-v1", learning_starts=100, verbose=0)
    model.learn(total_timesteps=1000)
    assert model.num_timesteps >= 1000
