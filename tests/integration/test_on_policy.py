"""Integration tests for on-policy algorithms (PPO, A2C)."""

from pathlib import Path
from typing import Any

import numpy as np


def _assert_eval_results(results, stage_prefix: str, expected_episodes: int) -> None:
    assert len(results) == 1
    assert results[0][f"{stage_prefix}/n_eval_episodes"] == expected_episodes
    assert f"{stage_prefix}/mean_reward" in results[0]
    assert f"{stage_prefix}/mean_episode_length" in results[0]


def _learn_with_tmpdir(model: Any, total_timesteps: int, env: str, tmp_path: Path) -> None:
    model.learn(total_timesteps=total_timesteps, env=env, default_root_dir=tmp_path)


def test_ppo_cartpole(tmp_path: Path) -> None:
    """PPO should run and complete training on CartPole-v1 for 1000 steps."""
    from lit_sb3.algorithms.ppo.ppo import PPO

    model = PPO("MlpPolicy", n_steps=64, batch_size=32, n_epochs=2, verbose=0)
    _learn_with_tmpdir(model, total_timesteps=1000, env="CartPole-v1", tmp_path=tmp_path)
    assert model.num_timesteps >= 1000


def test_a2c_cartpole(tmp_path: Path) -> None:
    """A2C should run and complete training on CartPole-v1 for 1000 steps."""
    from lit_sb3.algorithms.a2c.a2c import A2C

    model = A2C("MlpPolicy", n_steps=64, verbose=0)
    _learn_with_tmpdir(model, total_timesteps=1000, env="CartPole-v1", tmp_path=tmp_path)
    assert model.num_timesteps >= 1000


def test_ppo_predict(tmp_path: Path) -> None:
    """PPO policy should produce valid actions after training."""
    from lit_sb3.algorithms.ppo.ppo import PPO

    model = PPO("MlpPolicy", n_steps=64, verbose=0)
    _learn_with_tmpdir(model, total_timesteps=512, env="CartPole-v1", tmp_path=tmp_path)

    obs = model.env.reset()  # ty:ignore[unresolved-attribute]
    if isinstance(obs, tuple):
        obs = obs[0]
    action, _ = model.predict(obs, deterministic=True)
    assert action is not None


def test_ppo_save_load_preserves_deterministic_prediction(tmp_path: Path) -> None:
    """A saved PPO model should reload with identical deterministic actions."""
    from lit_sb3.algorithms.ppo.ppo import PPO

    model = PPO("MlpPolicy", n_steps=64, batch_size=32, n_epochs=2, verbose=0)
    _learn_with_tmpdir(model, total_timesteps=512, env="CartPole-v1", tmp_path=tmp_path)

    env = model.get_env()
    assert env is not None
    obs = env.reset()
    path = tmp_path / "ppo-checkpoint.zip"
    model.save(path)
    restored = PPO.load(path)

    expected_action, _ = model.predict(obs, deterministic=True)
    restored_action, _ = restored.predict(obs, deterministic=True)

    assert np.array_equal(restored_action, expected_action)
    assert restored.num_timesteps == model.num_timesteps


def test_ppo_cartpole_with_rltrainer(tmp_path: Path) -> None:
    """PPO should train and evaluate through RLTrainer using RLEnvDataModule stage envs."""
    from lit_sb3 import RLEnvDataModule, RLTrainer
    from lit_sb3.algorithms.ppo.ppo import PPO

    datamodule = RLEnvDataModule(
        train_env="CartPole-v1",
        val_env="CartPole-v1",
        test_env="CartPole-v1",
        predict_env="CartPole-v1",
    )
    trainer = RLTrainer(max_steps=512, default_root_dir=tmp_path)
    model = PPO("MlpPolicy", n_steps=64, batch_size=32, n_epochs=2, verbose=0)
    assert model.policy is None

    try:
        returned_model = trainer.fit(model, datamodule=datamodule)
        validation_results = trainer.validate(model, datamodule=datamodule, n_eval_episodes=2)
        test_results = trainer.test(model, datamodule=datamodule, n_eval_episodes=2)
        action, _ = trainer.predict(model, datamodule=datamodule)
    finally:
        datamodule.close()

    assert returned_model is model
    assert model.policy is not None
    assert model.num_timesteps >= 512
    assert trainer.rl_num_timesteps == model.num_timesteps
    assert trainer.rl_budget_reached is True
    _assert_eval_results(validation_results, "validate", expected_episodes=2)
    _assert_eval_results(test_results, "test", expected_episodes=2)
    assert np.asarray(action).size == 1


def test_a2c_cartpole_with_rltrainer(tmp_path: Path) -> None:
    """A2C should train and evaluate through RLTrainer using RLEnvDataModule stage envs."""
    from lit_sb3 import RLEnvDataModule, RLTrainer
    from lit_sb3.algorithms.a2c.a2c import A2C

    datamodule = RLEnvDataModule(
        train_env="CartPole-v1",
        val_env="CartPole-v1",
        test_env="CartPole-v1",
        predict_env="CartPole-v1",
    )
    trainer = RLTrainer(max_steps=512, default_root_dir=tmp_path)
    model = A2C("MlpPolicy", n_steps=64, verbose=0)

    try:
        returned_model = trainer.fit(model, datamodule=datamodule)
        validation_results = trainer.validate(model, datamodule=datamodule, n_eval_episodes=2)
        test_results = trainer.test(model, datamodule=datamodule, n_eval_episodes=2)
        action, _ = trainer.predict(model, datamodule=datamodule)
    finally:
        datamodule.close()

    assert returned_model is model
    assert model.num_timesteps >= 512
    assert trainer.rl_num_timesteps == model.num_timesteps
    assert trainer.rl_budget_reached is True
    _assert_eval_results(validation_results, "validate", expected_episodes=2)
    _assert_eval_results(test_results, "test", expected_episodes=2)
    assert np.asarray(action).size == 1
