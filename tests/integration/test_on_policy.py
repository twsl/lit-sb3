"""Integration tests for on-policy algorithms (PPO, A2C)."""

import pytest


def test_ppo_cartpole():
    """PPO should run and complete training on CartPole-v1 for 1000 steps."""
    from lit_sb3.algorithms.ppo import PPO

    model = PPO("MlpPolicy", "CartPole-v1", n_steps=64, batch_size=32, n_epochs=2, verbose=0)
    model.learn(total_timesteps=1000)
    assert model.num_timesteps >= 1000


def test_a2c_cartpole():
    """A2C should run and complete training on CartPole-v1 for 1000 steps."""
    from lit_sb3.algorithms.a2c import A2C

    model = A2C("MlpPolicy", "CartPole-v1", n_steps=64, verbose=0)
    model.learn(total_timesteps=1000)
    assert model.num_timesteps >= 1000


def test_ppo_predict():
    """PPO policy should produce valid actions after training."""
    import numpy as np

    from lit_sb3.algorithms.ppo import PPO

    model = PPO("MlpPolicy", "CartPole-v1", n_steps=64, verbose=0)
    model.learn(total_timesteps=512)

    obs = model.env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    action, _ = model.predict(obs, deterministic=True)
    assert action is not None
