"""Unit tests for RolloutDataset and ReplayBufferDataset."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


def test_replay_buffer_dataset_length():
    """ReplayBufferDataset should have correct length and yield integers."""
    from lit_sb3.data.replay import ReplayBufferDataset

    dataset = ReplayBufferDataset(total_steps=500)
    assert len(dataset) == 500

    items = list(dataset)
    assert len(items) == 500
    assert items[0] == 0
    assert items[-1] == 499


def test_replay_buffer_dataset_in_dataloader():
    """ReplayBufferDataset should work in a DataLoader."""
    from lit_sb3.data.replay import ReplayBufferDataset

    dataset = ReplayBufferDataset(total_steps=100)
    loader = DataLoader(dataset, batch_size=1)
    count = sum(1 for _ in loader)
    assert count == 100


def test_rollout_dataset_yields_batches():
    """RolloutDataset should yield batches from the rollout buffer."""
    import gymnasium as gym

    from lit_sb3.buffers import RolloutBuffer
    from lit_sb3.data.rollout import RolloutDataset

    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    act_space = env.action_space

    buffer = RolloutBuffer(
        buffer_size=64,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=1,
    )

    # Fill buffer with dummy data
    buffer.reset()
    for _i in range(64):
        obs = obs_space.sample()[None]  # add batch dim
        action = np.array([[0]])
        reward = np.array([1.0])
        episode_start = np.array([False])
        value = torch.zeros(1)
        log_prob = torch.zeros(1)
        buffer.add(obs, action, reward, episode_start, value, log_prob)

    buffer.compute_returns_and_advantage(last_values=torch.zeros(1), dones=np.array([False]))

    dataset = RolloutDataset(buffer, batch_size=16, n_epochs=2)
    batches = list(dataset)
    # 64 / 16 * 2 epochs = 8 batches
    assert len(batches) == 8
    for batch in batches:
        assert batch.observations.shape[0] == 16
