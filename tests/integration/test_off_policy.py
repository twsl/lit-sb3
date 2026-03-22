"""Integration tests for off-policy algorithms (SAC, TD3, DQN)."""

from pathlib import Path


def _assert_eval_results(results, stage_prefix: str, expected_episodes: int) -> None:
    assert len(results) == 1
    assert results[0][f"{stage_prefix}/n_eval_episodes"] == expected_episodes
    assert f"{stage_prefix}/mean_reward" in results[0]
    assert f"{stage_prefix}/mean_episode_length" in results[0]


def test_dqn_cartpole(tmp_path: Path) -> None:
    """DQN should run on CartPole-v1 for 1000 steps."""
    from lit_sb3.algorithms.dqn.dqn import DQN

    model = DQN("MlpPolicy", learning_starts=100, verbose=0)
    model.learn(total_timesteps=1000, env="CartPole-v1", default_root_dir=tmp_path)
    assert model.num_timesteps >= 1000


def test_sac_pendulum(tmp_path: Path) -> None:
    """SAC should run on Pendulum-v1 for 1000 steps."""
    from lit_sb3.algorithms.sac.sac import SAC

    model = SAC("MlpPolicy", learning_starts=100, verbose=0)
    model.learn(total_timesteps=1000, env="Pendulum-v1", default_root_dir=tmp_path)
    assert model.num_timesteps >= 1000


def test_td3_pendulum(tmp_path: Path) -> None:
    """TD3 should run on Pendulum-v1 for 1000 steps."""
    from lit_sb3.algorithms.td3.td3 import TD3

    model = TD3("MlpPolicy", learning_starts=100, verbose=0)
    model.learn(total_timesteps=1000, env="Pendulum-v1", default_root_dir=tmp_path)
    assert model.num_timesteps >= 1000


def test_dqn_cartpole_with_rltrainer(tmp_path: Path) -> None:
    """DQN should train and evaluate through RLTrainer using RLEnvDataModule stage envs."""
    from lit_sb3 import RLEnvDataModule, RLTrainer
    from lit_sb3.algorithms.dqn.dqn import DQN

    datamodule = RLEnvDataModule(train_env="CartPole-v1", test_env="CartPole-v1")
    trainer = RLTrainer(max_steps=512, default_root_dir=tmp_path)
    model = DQN("MlpPolicy", learning_starts=100, verbose=0)

    try:
        returned_model = trainer.fit(model, datamodule=datamodule)
        test_results = trainer.test(model, datamodule=datamodule, n_eval_episodes=2)
    finally:
        datamodule.close()

    assert returned_model is model
    assert model.num_timesteps >= 512
    assert trainer.rl_num_timesteps == model.num_timesteps
    assert trainer.rl_budget_reached is True
    _assert_eval_results(test_results, "test", expected_episodes=2)


def test_sac_pendulum_with_rltrainer(tmp_path: Path) -> None:
    """SAC should train and evaluate through RLTrainer using RLEnvDataModule stage envs."""
    from lit_sb3 import RLEnvDataModule, RLTrainer
    from lit_sb3.algorithms.sac.sac import SAC

    datamodule = RLEnvDataModule(train_env="Pendulum-v1", test_env="Pendulum-v1")
    trainer = RLTrainer(max_steps=512, default_root_dir=tmp_path)
    model = SAC("MlpPolicy", learning_starts=100, verbose=0)

    try:
        returned_model = trainer.fit(model, datamodule=datamodule)
        test_results = trainer.test(model, datamodule=datamodule, n_eval_episodes=2)
    finally:
        datamodule.close()

    assert returned_model is model
    assert model.num_timesteps >= 512
    assert trainer.rl_num_timesteps == model.num_timesteps
    assert trainer.rl_budget_reached is True
    _assert_eval_results(test_results, "test", expected_episodes=2)


def test_td3_pendulum_with_rltrainer(tmp_path: Path) -> None:
    """TD3 should train and evaluate through RLTrainer using RLEnvDataModule stage envs."""
    from lit_sb3 import RLEnvDataModule, RLTrainer
    from lit_sb3.algorithms.td3.td3 import TD3

    datamodule = RLEnvDataModule(train_env="Pendulum-v1", test_env="Pendulum-v1")
    trainer = RLTrainer(max_steps=512, default_root_dir=tmp_path)
    model = TD3("MlpPolicy", learning_starts=100, verbose=0)

    try:
        returned_model = trainer.fit(model, datamodule=datamodule)
        test_results = trainer.test(model, datamodule=datamodule, n_eval_episodes=2)
    finally:
        datamodule.close()

    assert returned_model is model
    assert model.num_timesteps >= 512
    assert trainer.rl_num_timesteps == model.num_timesteps
    assert trainer.rl_budget_reached is True
    _assert_eval_results(test_results, "test", expected_episodes=2)
