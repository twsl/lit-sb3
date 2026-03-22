"""RLTrainer — a ``lightning.Trainer`` subclass for reinforcement learning.

Usage::

    from lit_sb3 import PPO, RLEnvDataModule, RLTrainer

    model = PPO("MlpPolicy")
    dm = RLEnvDataModule(train_env="CartPole-v1")
    trainer = RLTrainer(max_steps=100_000, enable_progress_bar=True)
    trainer.fit(model, datamodule=dm)

The SB3-compatible ``model.learn(total_timesteps, env=...)`` API still works
and simply delegates to ``RLTrainer.fit`` internally.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import gymnasium as gym
import lightning as L  # noqa: N812
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning.fabric.utilities.cloud_io import _is_local_file_protocol
from lightning.fabric.utilities.types import _PATH
import lightning.pytorch as pl
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback, Checkpoint, EarlyStopping, ProgressBar
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.utilities import _log_hyperparams
from lightning.pytorch.loops import _PredictionLoop, _TrainingEpochLoop
from lightning.pytorch.loops.evaluation_loop import _EvaluationLoop
from lightning.pytorch.loops.fit_loop import _FitLoop
from lightning.pytorch.loops.utilities import _parse_loop_limits, _reset_progress
from lightning.pytorch.plugins import _PLUGIN_INPUT, Precision
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import ParallelStrategy, Strategy
from lightning.pytorch.trainer import call, setup
from lightning.pytorch.trainer.configuration_validator import _verify_loop_configurations
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
    _PRECISION_INPUT_STR,
    _AcceleratorConnector,
)
from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector
from lightning.pytorch.trainer.connectors.checkpoint_connector import _CheckpointConnector
from lightning.pytorch.trainer.connectors.data_connector import _DataConnector
from lightning.pytorch.trainer.connectors.logger_connector import _LoggerConnector
from lightning.pytorch.trainer.connectors.logger_connector.result import _OUT_DICT, _PBAR_DICT, _ResultCollection
from lightning.pytorch.trainer.connectors.signal_connector import _SignalConnector
from lightning.pytorch.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from lightning.pytorch.utilities import GradClipAlgorithmType, parsing
from lightning.pytorch.utilities.argparse import _defaults_from_env_vars
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized, _verify_strategy_supports_compile
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.model_registry import _is_registry, download_model_from_registry
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
    LRSchedulerConfig,
)
from lightning.pytorch.utilities.warnings import PossibleUserWarning
import numpy as np

from lit_sb3.callbacks.summary_writer import SummaryWriterCallback
from lit_sb3.core.evaluation import evaluate_policy
from lit_sb3.core.types import GymEnv, MaybeCallback
from lit_sb3.utils.system import get_latest_run_id

if TYPE_CHECKING:
    from lit_sb3.algorithms.base import BaseAlgorithm


class RLTrainer(L.Trainer):
    """A :class:`lightning.Trainer` subclass that drives reinforcement-learning runs.

    All standard ``lightning.Trainer`` keyword arguments are accepted and
    forwarded to the parent class, giving full access to Lightning features
    (mixed precision, custom loggers, DDP, plugins, …).

    The RL-specific arguments control the training schedule and SB3 callbacks.
    ``accelerator``, ``enable_model_summary``, ``enable_checkpointing``, and
    ``logger`` have RL-appropriate defaults but can be overridden.

    Args:
        max_steps: Total environment steps to collect (RL timesteps budget).
            Lightning step-based stopping is disabled; this budget is enforced
            using RL env-step accounting instead.
        rl_callback: SB3-style callback(s) fired at each RL step.
        run_name: The run name used in loggers.
    """

    def __init__(
        self,
        max_steps: int,
        rl_callback: MaybeCallback = None,
        run_name: str = "run",
        *,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: list[int] | str | int = "auto",
        num_nodes: int = 1,
        precision: _PRECISION_INPUT | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        callbacks: list[Callback] | Callback | None = None,
        fast_dev_run: int | bool = False,
        max_epochs: int | None = None,
        min_epochs: int | None = None,
        # max_steps: int = -1,
        min_steps: int | None = None,
        max_time: str | timedelta | dict[str, int] | None = None,
        limit_train_batches: int | float | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        limit_predict_batches: int | float | None = None,
        overfit_batches: int | float = 0.0,
        val_check_interval: int | float | str | timedelta | dict[str, int] | None = None,
        check_val_every_n_epoch: int | None = 1,
        num_sanity_val_steps: int | None = None,
        log_every_n_steps: int | None = None,
        enable_checkpointing: bool | None = None,
        enable_progress_bar: bool | None = None,
        enable_model_summary: bool | None = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
        deterministic: bool | _LITERAL_WARN | None = None,
        benchmark: bool | None = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Profiler | str | None = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: _PLUGIN_INPUT | list[_PLUGIN_INPUT] | None = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: _PATH | None = None,
        enable_autolog_hparams: bool = True,
        model_registry: str | None = None,
    ) -> None:
        self._validate_trainer_init(max_epochs, min_epochs, max_steps, min_steps)

        self._rl_total_timesteps = max_steps
        self._rl_callback = rl_callback
        self._rl_log_every_n_steps = log_every_n_steps
        self._rl_run_name = run_name
        self._rl_reset_num_timesteps = True
        self._rl_env: Any = None  # Active training env; set by fit()
        self._rl_num_timesteps = 0
        self._rl_budget_reached = False
        self._rl_auto_logger = logger is None
        trainer_logger = False if self._rl_auto_logger else logger

        super().__init__(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            logger=trainer_logger,
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=-1,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            overfit_batches=overfit_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            deterministic=deterministic,
            benchmark=benchmark,
            inference_mode=inference_mode,
            use_distributed_sampler=use_distributed_sampler,
            profiler=profiler,
            detect_anomaly=detect_anomaly,
            barebones=barebones,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            default_root_dir=default_root_dir,
            enable_autolog_hparams=enable_autolog_hparams,
            model_registry=model_registry,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def rl_total_timesteps(self) -> int:
        """Return the trainer-owned RL timestep budget."""
        return self._rl_total_timesteps

    @property
    def rl_num_timesteps(self) -> int:
        """Return the latest observed RL env-step count."""
        return self._rl_num_timesteps

    @property
    def rl_budget_reached(self) -> bool:
        """Return whether the trainer-owned RL timestep budget has been reached."""
        return self._rl_budget_reached

    @property
    def rl_progress(self) -> float:
        """Return RL budget progress as a value in ``[0, 1]``."""
        if self._rl_total_timesteps <= 0:
            return 0.0
        return min(1.0, self._rl_num_timesteps / self._rl_total_timesteps)

    def _reset_rl_budget_tracking(self) -> None:
        self._rl_num_timesteps = 0
        self._rl_budget_reached = False

    def _update_rl_timesteps(self, num_timesteps: int) -> None:
        self._rl_num_timesteps = max(0, int(num_timesteps))
        self._rl_budget_reached = self._rl_num_timesteps >= self._rl_total_timesteps

    @staticmethod
    def _validate_trainer_init(
        max_epochs: int | None,
        min_epochs: int | None,
        max_steps: int,
        min_steps: int | None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("RLTrainer requires `max_steps` to be a positive integer.")
        unsupported = []
        if max_epochs is not None:
            unsupported.append("max_epochs")
        if min_epochs is not None:
            unsupported.append("min_epochs")
        # if max_steps != -1:
        #     unsupported.append("max_steps")
        if min_steps is not None:
            unsupported.append("min_steps")
        if unsupported:
            names = ", ".join(unsupported)
            raise ValueError(
                f"RLTrainer manages Lightning loop limits from `max_steps`; unsupported constructor args: {names}."
            )

    @staticmethod
    def _reject_unsupported_kwargs(stage: str, **kwargs: Any) -> None:
        provided = [name for name, value in kwargs.items() if value is not None]
        if provided:
            joined = ", ".join(provided)
            raise TypeError(f"RLTrainer.{stage}() does not support Lightning data/checkpoint inputs ({joined}).")

    def _resolve_model(self, model: BaseAlgorithm | None) -> BaseAlgorithm:
        resolved = model if model is not None else self.lightning_module
        if resolved is None:
            raise ValueError("No model provided and no model is attached to this trainer.")
        if not isinstance(resolved, L.LightningModule):
            raise TypeError("RLTrainer expects a lit-sb3 algorithm model.")
        return resolved  # type: ignore[return-value]

    @staticmethod
    def _reset_env(env: gym.Env | Any) -> Any:
        observation = env.reset()
        if isinstance(observation, tuple):
            return observation[0]
        return observation

    def _resolve_eval_env(self, model: BaseAlgorithm, env: gym.Env | Any | None) -> gym.Env | Any:
        resolved_env = env if env is not None else self._rl_env
        if resolved_env is None:
            resolved_env = model.get_env()
        if resolved_env is None:
            raise ValueError("No evaluation environment was provided and no environment is attached to the trainer.")
        return resolved_env

    def _evaluate(
        self,
        stage: str,
        model: BaseAlgorithm | None,
        *,
        env: gym.Env | Any | None = None,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        reward_threshold: float | None = None,
        return_episode_rewards: bool = False,
        use_masking: bool = False,
    ) -> list[Mapping[str, float | int | list[float] | list[int]]]:
        resolved_model = self._resolve_model(model)
        eval_env = self._resolve_eval_env(resolved_model, env)
        episode_rewards, episode_lengths = evaluate_policy(
            resolved_model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            reward_threshold=reward_threshold,
            return_episode_rewards=True,
            use_masking=use_masking,
        )
        typed_rewards = cast(list[float], episode_rewards)
        typed_lengths = cast(list[int], episode_lengths)

        result: dict[str, float | int | list[float] | list[int]] = {
            f"{stage}/mean_reward": float(np.mean(typed_rewards)),
            f"{stage}/std_reward": float(np.std(typed_rewards)),
            f"{stage}/mean_episode_length": float(np.mean(typed_lengths)),
            f"{stage}/std_episode_length": float(np.std(typed_lengths)),
            f"{stage}/n_eval_episodes": int(len(typed_rewards)),
        }
        if return_episode_rewards:
            result[f"{stage}/episode_rewards"] = [float(reward) for reward in typed_rewards]
            result[f"{stage}/episode_lengths"] = [int(length) for length in typed_lengths]
        # print(
        #         f"{stage}: mean_reward={result[f'{stage}/mean_reward']:.2f}, "
        #         f"std_reward={result[f'{stage}/std_reward']:.2f}, "
        #         f"mean_episode_length={result[f'{stage}/mean_episode_length']:.2f}"
        #     )
        return [result]

    def _attach_rl_callback(self, rl_callback: Any) -> None:
        callbacks = cast(list[Any], getattr(self, "callbacks", []))
        if rl_callback not in callbacks:
            callbacks.append(rl_callback)

    def _ensure_summary_writer_callback(self) -> None:
        callbacks = cast(list[Any], getattr(self, "callbacks", []))
        if any(isinstance(callback, SummaryWriterCallback) for callback in callbacks):
            return
        callbacks.append(SummaryWriterCallback())

    def _set_runtime_logger(self, logger: Any) -> None:
        connector = getattr(self, "_logger_connector", None)
        if connector is not None:
            configure_logger = getattr(connector, "configure_logger", None)
            if callable(configure_logger):
                configure_logger(logger)
            elif hasattr(connector, "_loggers"):
                connector._loggers = [] if logger in (None, False) else [logger]
        self._loggers = [] if logger in (None, False) else [logger]

    def _configure_lightning_logger(self, model: BaseAlgorithm) -> None:
        existing_loggers = [logger for logger in getattr(self, "loggers", []) if logger not in (None, False)]
        if existing_loggers:
            self._ensure_summary_writer_callback()
            return

        logger = model._trainer_logger
        if logger is None and self._rl_auto_logger:
            logger_root = Path(self.default_root_dir) / "lightning_logs"
            latest_run_id = get_latest_run_id(logger_root, self._rl_run_name)
            run_id = latest_run_id + 1 if self._rl_reset_num_timesteps or latest_run_id == 0 else latest_run_id
            logger = TensorBoardLogger(
                save_dir=str(self.default_root_dir),
                name="lightning_logs",
                version=f"{self._rl_run_name}_{run_id}",
                default_hp_metric=False,
            )

        if logger is not None:
            self._set_runtime_logger(logger)
        self._ensure_summary_writer_callback()

    @staticmethod
    def _resolve_datamodule_env(datamodule: Any, stage: Literal["fit", "validate", "test", "predict"]) -> Any:
        from lit_sb3.datamodule import RLEnvDataModule

        if not isinstance(datamodule, RLEnvDataModule):
            raise TypeError(f"RLTrainer expects an RLEnvDataModule, got {type(datamodule).__name__}.")
        return datamodule.get_env(stage)

    def _log_runtime_device(self, model: BaseAlgorithm) -> None:
        if model.verbose < 1:
            return
        strategy = getattr(self, "strategy", None)
        root_device = getattr(strategy, "root_device", None)
        if root_device is not None:
            print(f"Using {root_device} device")

    def fit(  # type: ignore[override]
        self,
        model: BaseAlgorithm,
        *,
        datamodule: Any = None,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        ckpt_path: str | Path | None = None,
        weights_only: bool | None = None,
    ) -> BaseAlgorithm:
        """Train *model* for :attr:`_rl_total_timesteps` environment steps.

        The environment is provided via an :class:`~lit_sb3.datamodule.RLEnvDataModule`.

        Args:
            model: The RL algorithm (a ``LightningModule`` subclass).
            datamodule: An :class:`~lit_sb3.datamodule.RLEnvDataModule` that provides
                the training (and optionally evaluation) environments.
            train_dataloaders: Unsupported for RLTrainer.
            val_dataloaders: Unsupported for RLTrainer.
            ckpt_path: Optional Lightning checkpoint path.
            weights_only: Optional Lightning checkpoint loading flag.

        Returns:
            The trained model (same object, for chaining).
        """
        from lit_sb3.vec_env import unwrap_vec_normalize

        self._reject_unsupported_kwargs(
            "fit",
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
        )

        # --- Resolve the training environment from the datamodule ---
        if datamodule is None:
            raise TypeError(
                "RLTrainer.fit() requires a datamodule (RLEnvDataModule) that provides the training environment."
            )
        resolved_env = self._resolve_datamodule_env(datamodule, "fit")
        if resolved_env is None:
            raise ValueError(
                "The RLEnvDataModule has no train_env. Provide train_env when constructing the RLEnvDataModule."
            )

        self._rl_env = resolved_env
        self._reset_rl_budget_tracking()

        # Ensure the model has env metadata (observation_space, action_space, n_envs)
        if not hasattr(model, "observation_space") or not hasattr(model, "n_envs"):
            model._init_env_metadata(
                resolved_env,
                getattr(model, "_supported_action_spaces", None),
                getattr(model, "_support_multi_env", False),
            )
        model._vec_normalize_env = unwrap_vec_normalize(resolved_env)

        # Give model a reference to this trainer so env property resolves
        # before Lightning's super().fit() formally attaches it.
        model._rl_trainer_ref = self  # type: ignore[attr-defined]
        self._log_runtime_device(model)

        # If the policy has not been built yet (deferred setup), build it now.
        import torch.nn as nn

        if not isinstance(getattr(model, "policy", None), nn.Module):
            model._setup_model()

        self._configure_lightning_logger(model)

        total_timesteps, rl_callback = model._setup_learn(
            self._rl_total_timesteps,
            self._rl_callback,
            self._rl_run_name,
        )
        self._update_rl_timesteps(model.num_timesteps)

        log_every_n_steps = (
            self._rl_log_every_n_steps if self._rl_log_every_n_steps is not None else model._default_log_every_n_steps
        )

        model._prepare_fit(rl_callback, log_every_n_steps)

        # Patch max_epochs so Lightning runs the right number of rollout cycles.
        self.fit_loop.max_epochs = model._rl_max_epochs(total_timesteps)

        # Attach the RL callback once so its Lightning hooks participate in fit.
        self._attach_rl_callback(rl_callback)

        # Algorithm-specific pre-fit work (on-policy: pre-collect first rollout).
        rl_callback.on_training_start(locals(), globals())
        model._pre_fit_hook(rl_callback)

        super().fit(
            model,
            train_dataloaders=None,
            val_dataloaders=None,
            datamodule=None,
            ckpt_path=ckpt_path,
            weights_only=weights_only,
        )
        self._update_rl_timesteps(model.num_timesteps)
        # on_train_end() in the model calls rl_callback.on_training_end()
        return model

    def validate(  # type: ignore[override]
        self,
        model: BaseAlgorithm | None = None,
        dataloaders: Any = None,
        ckpt_path: str | Path | None = None,
        datamodule: Any = None,
        weights_only: bool | None = None,
        *,
        env: gym.Env | Any | None = None,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        reward_threshold: float | None = None,
        return_episode_rewards: bool = False,
        use_masking: bool = False,
    ) -> list[Mapping[str, float | int | list[float] | list[int]]]:
        self._reject_unsupported_kwargs(
            "validate",
            dataloaders=dataloaders,
            ckpt_path=ckpt_path,
            weights_only=weights_only,
        )
        # Resolve env from datamodule if not passed directly
        if env is None and datamodule is not None:
            env = self._resolve_datamodule_env(datamodule, "validate")
        return self._evaluate(
            "validate",
            model,
            env=env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            reward_threshold=reward_threshold,
            return_episode_rewards=return_episode_rewards,
            use_masking=use_masking,
        )

    def test(  # type: ignore[override]
        self,
        model: BaseAlgorithm | None = None,
        dataloaders: Any = None,
        ckpt_path: str | Path | None = None,
        datamodule: Any = None,
        weights_only: bool | None = None,
        *,
        env: gym.Env | Any | None = None,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        reward_threshold: float | None = None,
        return_episode_rewards: bool = False,
        use_masking: bool = False,
    ) -> list[Mapping[str, float | int | list[float] | list[int]]]:
        self._reject_unsupported_kwargs(
            "test",
            dataloaders=dataloaders,
            ckpt_path=ckpt_path,
            weights_only=weights_only,
        )
        # Resolve env from datamodule if not passed directly
        if env is None and datamodule is not None:
            env = self._resolve_datamodule_env(datamodule, "test")
        return self._evaluate(
            "test",
            model,
            env=env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            reward_threshold=reward_threshold,
            return_episode_rewards=return_episode_rewards,
            use_masking=use_masking,
        )

    def predict(  # type: ignore[override]
        self,
        model: BaseAlgorithm | None = None,
        dataloaders: Any = None,
        datamodule: Any = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
        weights_only: bool | None = None,
        *,
        observation: np.ndarray | dict[str, np.ndarray] | None = None,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
        env: gym.Env | Any | None = None,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        self._reject_unsupported_kwargs(
            "predict",
            dataloaders=dataloaders,
            ckpt_path=ckpt_path,
            weights_only=weights_only,
        )
        # Resolve env from datamodule if not passed directly
        if env is None and datamodule is not None:
            env = self._resolve_datamodule_env(datamodule, "predict")
        if return_predictions is False:
            raise TypeError("RLTrainer.predict() always returns policy predictions.")

        resolved_model = self._resolve_model(model)
        if observation is None:
            prediction_env = self._resolve_eval_env(resolved_model, env)
            observation = self._reset_env(prediction_env)

        if action_masks is None:
            return resolved_model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
            )

        predict_fn = cast(Any, resolved_model.predict)
        return predict_fn(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )
