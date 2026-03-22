from pathlib import Path
from unittest.mock import patch

from lightning.pytorch.loggers import TensorBoardLogger
import pytest

from lit_sb3 import PPO, RLTrainer
from lit_sb3.callbacks import BaseCallback, SummaryWriterCallback
from lit_sb3.datamodule import RLEnvDataModule


class CountingCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.fit_start_calls = 0
        self.fit_end_calls = 0
        self.training_start_calls = 0
        self.training_end_calls = 0

    def on_fit_start(self, trainer, pl_module) -> None:
        self.fit_start_calls += 1
        super().on_fit_start(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module) -> None:
        self.fit_end_calls += 1
        super().on_fit_end(trainer, pl_module)

    def _on_training_start(self) -> None:
        self.training_start_calls += 1

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self.training_end_calls += 1


def make_ppo() -> PPO:
    return PPO("MlpPolicy", n_steps=32, batch_size=32, n_epochs=1, verbose=0)


def make_dm(env: str = "CartPole-v1") -> RLEnvDataModule:
    return RLEnvDataModule(train_env=env)


def make_trainer(tmp_path: Path, **kwargs: object) -> RLTrainer:
    return RLTrainer(max_steps=32, default_root_dir=tmp_path, **kwargs)  # ty:ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("fit_kwargs", "expected_fragment"),
    [
        ({"train_dataloaders": [1]}, "train_dataloaders"),
        ({"val_dataloaders": [1]}, "val_dataloaders"),
    ],
)
def test_rltrainer_rejects_external_data_sources(
    tmp_path: Path,
    fit_kwargs,
    expected_fragment,
) -> None:
    model = make_ppo()
    trainer = make_trainer(tmp_path)

    with pytest.raises(TypeError, match=expected_fragment):
        trainer.fit(model, datamodule=make_dm(), **fit_kwargs)


def test_rltrainer_rejects_non_RLEnvDataModule(tmp_path: Path) -> None:  # noqa: N802
    model = make_ppo()
    trainer = make_trainer(tmp_path)

    with pytest.raises(TypeError, match="RLEnvDataModule"):
        trainer.fit(model, datamodule=object())


def test_rltrainer_requires_datamodule(tmp_path: Path) -> None:
    model = make_ppo()
    trainer = make_trainer(tmp_path)

    with pytest.raises(TypeError, match="datamodule"):
        trainer.fit(model)


def test_rltrainer_reuses_callback_cleanly_across_fits(tmp_path: Path) -> None:
    callback = CountingCallback()

    first_model = make_ppo()
    first_trainer = make_trainer(tmp_path / "first", rl_callback=callback)
    returned_first_model = first_trainer.fit(first_model, datamodule=make_dm())

    second_model = make_ppo()
    second_trainer = make_trainer(tmp_path / "second", rl_callback=callback)
    returned_second_model = second_trainer.fit(second_model, datamodule=make_dm())

    assert returned_first_model is first_model
    assert returned_second_model is second_model
    assert callback.fit_start_calls == 2
    assert callback.fit_end_calls == 2
    assert callback.training_start_calls == 2
    assert callback.training_end_calls == 2
    assert sum(candidate is callback for candidate in first_trainer.callbacks) == 1  # ty:ignore[unresolved-attribute]
    assert sum(candidate is callback for candidate in second_trainer.callbacks) == 1  # ty:ignore[unresolved-attribute]


def test_rltrainer_enables_default_logger_when_omitted(tmp_path: Path) -> None:
    model = make_ppo()
    trainer = make_trainer(tmp_path)
    datamodule = make_dm()

    try:
        trainer.fit(model, datamodule=datamodule)
    finally:
        datamodule.close()

    assert len(trainer.loggers) == 1
    assert isinstance(trainer.loggers[0], TensorBoardLogger)
    assert trainer.loggers[0].save_dir == str(tmp_path)
    assert trainer.loggers[0].name == "lightning_logs"
    assert trainer.loggers[0].version == "run_1"
    assert trainer.loggers[0].log_dir == str(tmp_path / "lightning_logs" / "run_1")


def test_rltrainer_respects_explicit_logger_disable(tmp_path: Path) -> None:
    model = make_ppo()
    trainer = make_trainer(tmp_path, logger=False)
    datamodule = make_dm()

    try:
        trainer.fit(model, datamodule=datamodule)
    finally:
        datamodule.close()

    assert trainer.loggers == []


def test_model_device_uses_rltrainer_root_device(tmp_path: Path) -> None:
    model = make_ppo()
    trainer = make_trainer(tmp_path)

    model._rl_trainer_ref = trainer  # type: ignore[attr-defined]

    assert model.device == trainer.strategy.root_device


def test_create_trainer_returns_rltrainer_with_learn_settings() -> None:
    model = make_ppo()
    callback = CountingCallback()

    trainer = model.create_trainer(
        max_steps=128,
        callback=callback,
        log_every_n_steps=7,
        run_name="PPO",
        reset_num_timesteps=False,
        enable_progress_bar=True,
    )

    assert isinstance(trainer, RLTrainer)
    assert trainer._rl_total_timesteps == 128
    assert trainer._rl_callback is callback
    assert trainer._rl_log_every_n_steps == 7
    assert trainer._rl_run_name == "PPO"
    assert trainer._rl_reset_num_timesteps is False
    # assert trainer.enable_progress_bar is True


def test_rltrainer_keeps_rl_budget_separate_from_lightning_max_steps(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)

    assert trainer.rl_total_timesteps == 32
    assert trainer.max_steps == -1
    assert trainer.rl_num_timesteps == 0
    assert trainer.rl_budget_reached is False
    assert trainer.rl_progress == 0.0

    trainer._update_rl_timesteps(16)
    assert trainer.rl_num_timesteps == 16
    assert trainer.rl_budget_reached is False
    assert trainer.rl_progress == pytest.approx(0.5)

    trainer._update_rl_timesteps(32)
    assert trainer.rl_num_timesteps == 32
    assert trainer.rl_budget_reached is True
    assert trainer.rl_progress == pytest.approx(1.0)


def test_rltrainer_configures_default_model_logger_during_fit(tmp_path: Path) -> None:
    model = PPO(
        "MlpPolicy",
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        verbose=1,
    )
    trainer = make_trainer(tmp_path / "trainer-root", run_name="trainer-fit")
    datamodule = make_dm()

    try:
        trainer.fit(model, datamodule=datamodule)
    finally:
        datamodule.close()

    assert len(trainer.loggers) == 1
    assert isinstance(trainer.loggers[0], TensorBoardLogger)
    assert trainer.loggers[0].save_dir == str(tmp_path / "trainer-root")
    assert trainer.loggers[0].name == "lightning_logs"
    assert trainer.loggers[0].version == "trainer-fit_1"
    assert trainer.loggers[0].log_dir == str(tmp_path / "trainer-root" / "lightning_logs" / "trainer-fit_1")
    assert any(isinstance(callback, SummaryWriterCallback) for callback in trainer.callbacks)  # ty:ignore[unresolved-attribute]
    assert model.get_summary_writer() is not None


def test_rltrainer_does_not_overwrite_custom_model_logger(tmp_path: Path) -> None:
    model = make_ppo()
    custom_logger = TensorBoardLogger(save_dir=str(tmp_path), name="custom", version="fixed")
    model.set_logger(custom_logger)

    trainer = make_trainer(tmp_path / "trainer-root", run_name="trainer-fit")
    datamodule = make_dm()

    try:
        trainer.fit(model, datamodule=datamodule)
    finally:
        datamodule.close()

    assert len(trainer.loggers) == 1
    assert trainer.loggers[0] is custom_logger
    assert any(isinstance(callback, SummaryWriterCallback) for callback in trainer.callbacks)  # ty:ignore[unresolved-attribute]
    assert model.get_summary_writer() is not None


@pytest.mark.parametrize(
    ("method_name", "stage", "attribute_name"),
    [
        ("validate", "validate", "val_env"),
        ("test", "test", "test_env"),
        ("predict", "predict", "predict_env"),
    ],
)
def test_rltrainer_uses_datamodule_stage_env(
    tmp_path: Path,
    method_name,
    stage,
    attribute_name,
) -> None:
    trainer = make_trainer(tmp_path)
    datamodule = RLEnvDataModule()
    resolved_env = object()

    setattr(datamodule, attribute_name, resolved_env)

    with patch.object(datamodule, "get_env", return_value=resolved_env) as mock_get_env:
        if method_name == "predict":
            model = make_ppo()
            with (
                patch.object(trainer, "_resolve_model", return_value=model),
                patch.object(trainer, "_resolve_eval_env", return_value=resolved_env),
                patch.object(trainer, "_reset_env", return_value="obs"),
                patch.object(model, "predict", return_value=("action", None)) as mock_predict,
            ):
                result = trainer.predict(model, datamodule=datamodule)
            assert result == ("action", None)
            mock_predict.assert_called_once_with("obs", state=None, episode_start=None, deterministic=True)
        else:
            with patch.object(trainer, "_evaluate", return_value=[{"ok": 1}]) as mock_evaluate:
                result = getattr(trainer, method_name)(datamodule=datamodule)
            assert result == [{"ok": 1}]
            mock_evaluate.assert_called_once()
            assert mock_evaluate.call_args.kwargs["env"] is resolved_env

    mock_get_env.assert_called_once_with(stage)
