from __future__ import annotations

from dataclasses import dataclass

import pytest

from lit_sb3.datamodule import RLEnvDataModule


@dataclass
class FakeVecEnv:
    name: str
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def test_get_env_resolves_requested_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    dm = RLEnvDataModule(train_env="train-id", val_env="val-id", test_env="test-id", predict_env="predict-id")

    monkeypatch.setattr(dm, "_resolve", lambda spec: FakeVecEnv(str(spec)))

    train_env = dm.get_env("fit")
    val_env = dm.get_env("validate")
    test_env = dm.get_env("test")
    predict_env = dm.get_env("predict")

    assert train_env is not None and train_env.name == "train-id"
    assert val_env is not None and val_env.name == "val-id"
    assert test_env is not None and test_env.name == "test-id"
    assert predict_env is not None and predict_env.name == "predict-id"


def test_state_dict_persists_only_serializable_specs() -> None:
    dm = RLEnvDataModule(train_env="CartPole-v1", val_env=object(), test_env=None, predict_env="Pendulum-v1", verbose=3)

    state = dm.state_dict()

    assert state["train_env_spec"] == {"kind": "string", "value": "CartPole-v1"}
    assert state["val_env_spec"] == {"kind": "unserializable", "value": None}
    assert state["test_env_spec"] == {"kind": "none", "value": None}
    assert state["predict_env_spec"] == {"kind": "string", "value": "Pendulum-v1"}
    assert state["monitor_wrapper"] is True
    assert state["verbose"] == 3


def test_load_state_dict_closes_old_envs_and_restores_specs() -> None:
    dm = RLEnvDataModule(train_env="before")
    dm.train_env = FakeVecEnv("old-train")
    dm.val_env = FakeVecEnv("old-val")

    dm.load_state_dict(
        {
            "train_env_spec": {"kind": "string", "value": "CartPole-v1"},
            "val_env_spec": {"kind": "string", "value": "Acrobot-v1"},
            "test_env_spec": {"kind": "none", "value": None},
            "predict_env_spec": {"kind": "unserializable", "value": None},
            "monitor_wrapper": False,
            "verbose": 7,
        }
    )

    assert dm.train_env is None
    assert dm.val_env is None
    assert dm._train_env_spec == "CartPole-v1"
    assert dm._val_env_spec == "Acrobot-v1"
    assert dm._test_env_spec is None
    assert dm._predict_env_spec is None
    assert dm._monitor_wrapper is False
    assert dm._verbose == 7


def test_teardown_closes_stage_specific_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    dm = RLEnvDataModule(train_env="train-id", val_env="val-id", test_env="test-id")

    monkeypatch.setattr(dm, "_resolve", lambda spec: FakeVecEnv(str(spec)))
    dm.setup("fit")
    dm.setup("test")

    train_env = dm.train_env
    val_env = dm.val_env
    test_env = dm.test_env

    dm.teardown("fit")

    assert train_env is not None and train_env.closed is True
    assert val_env is not None and val_env.closed is True
    assert test_env is not None and test_env.closed is False
    assert dm.train_env is None
    assert dm.val_env is None
    assert dm.test_env is test_env


def test_dataloader_methods_raise_clear_errors() -> None:
    dm = RLEnvDataModule(train_env="CartPole-v1")

    with pytest.raises(TypeError, match="train_dataloader"):
        dm.train_dataloader()
    with pytest.raises(TypeError, match="val_dataloader"):
        dm.val_dataloader()
    with pytest.raises(TypeError, match="test_dataloader"):
        dm.test_dataloader()
    with pytest.raises(TypeError, match="predict_dataloader"):
        dm.predict_dataloader()


def test_str_reports_env_specs_and_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    dm = RLEnvDataModule(train_env="CartPole-v1", predict_env="Pendulum-v1")
    monkeypatch.setattr(dm, "_resolve", lambda spec: FakeVecEnv(str(spec)))

    dm.setup("fit")
    rendered = str(dm)

    assert "{fit: spec=CartPole-v1, status=resolved, env=FakeVecEnv}" in rendered
    assert "{validate: spec=None, status=unresolved, env=None}" in rendered
    assert "{predict: spec=Pendulum-v1, status=unresolved, env=None}" in rendered
