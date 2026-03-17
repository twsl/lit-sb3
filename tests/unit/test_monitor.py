"""Unit tests for the Monitor wrapper.

Adapted from HenryJia/lightning-baselines3 tests/common/test_monitor.py.
"""

import json
import os
import uuid

import gymnasium as gym
import pandas

from lit_sb3.common.monitor import Monitor, get_monitor_files, load_results


def test_monitor(tmp_path):
    """Test the Monitor wrapper records steps, rewards, and episode metadata."""
    env = gym.make("CartPole-v1")
    monitor_file = os.path.join(str(tmp_path), f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env = Monitor(env, monitor_file)
    monitor_env.reset()
    total_steps = 1000
    ep_rewards = []
    ep_lengths = []
    ep_len, ep_reward = 0, 0.0
    for _ in range(total_steps):
        _, reward, terminated, truncated, _ = monitor_env.step(monitor_env.action_space.sample())
        ep_len += 1
        ep_reward += float(reward)
        if terminated or truncated:
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)
            monitor_env.reset()
            ep_len, ep_reward = 0, 0.0
    monitor_env.close()
    assert monitor_env.get_total_steps() == total_steps
    assert sum(ep_lengths) == sum(monitor_env.get_episode_lengths())
    assert sum(monitor_env.get_episode_rewards()) == sum(ep_rewards)
    _ = monitor_env.get_episode_times()

    with open(monitor_file) as file_handler:
        first_line = file_handler.readline()
        assert first_line.startswith("#")
        metadata = json.loads(first_line[1:])
        assert metadata["env_id"] == "CartPole-v1"
        assert set(metadata.keys()) == {"env_id", "t_start"}, "Incorrect keys in monitor metadata"

        last_logline = pandas.read_csv(file_handler, index_col=None)
        assert set(last_logline.keys()) == {"l", "t", "r"}, "Incorrect keys in monitor logline"
    os.remove(monitor_file)


def test_monitor_load_results(tmp_path):
    """Test load_results on log files produced by the Monitor wrapper."""
    tmp_path = str(tmp_path)
    env1 = gym.make("CartPole-v1")
    monitor_file1 = os.path.join(tmp_path, f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env1 = Monitor(env1, monitor_file1)

    monitor_files = get_monitor_files(tmp_path)
    assert len(monitor_files) == 1
    assert monitor_file1 in monitor_files

    monitor_env1.reset()
    episode_count1 = 0
    for _ in range(1000):
        _, _, terminated, truncated, _ = monitor_env1.step(monitor_env1.action_space.sample())
        if terminated or truncated:
            episode_count1 += 1
            monitor_env1.reset()

    results_size1 = len(load_results(tmp_path).index)
    assert results_size1 == episode_count1

    env2 = gym.make("CartPole-v1")
    monitor_file2 = os.path.join(tmp_path, f"stable_baselines-test-{uuid.uuid4()}.monitor.csv")
    monitor_env2 = Monitor(env2, monitor_file2)

    monitor_files = get_monitor_files(tmp_path)
    assert len(monitor_files) == 2
    assert monitor_file1 in monitor_files
    assert monitor_file2 in monitor_files

    monitor_env2.reset()
    episode_count2 = 0
    for _ in range(1000):
        _, _, terminated, truncated, _ = monitor_env2.step(monitor_env2.action_space.sample())
        if terminated or truncated:
            episode_count2 += 1
            monitor_env2.reset()

    results_size2 = len(load_results(tmp_path).index)
    assert results_size2 == (results_size1 + episode_count2)

    os.remove(monitor_file1)
    os.remove(monitor_file2)
