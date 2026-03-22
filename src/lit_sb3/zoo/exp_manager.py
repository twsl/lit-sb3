"""ExperimentManager — reads hyperparameters, creates envs, drives training.

Adapted from
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/exp_manager.py
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import importlib
import os
from pathlib import Path
import pickle as pkl
from pprint import pprint
import sys
import time
from typing import Any
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
import yaml

from lit_sb3.algorithms.base import BaseAlgorithm
from lit_sb3.algorithms.her.her_replay_buffer import HerReplayBuffer
from lit_sb3.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
    SaveVecNormalizeCallback,
    TrialEvalCallback,
)
from lit_sb3.core.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from lit_sb3.core.preprocessing import is_image_space, is_image_space_channels_first
from lit_sb3.huggingface.naming_schemes import EnvironmentName
from lit_sb3.utils.env import make_vec_env
from lit_sb3.utils.reflection import get_callback_list, get_class_by_name, get_wrapper_class
from lit_sb3.utils.registry import ALGOS
from lit_sb3.utils.schedule import ConstantSchedule, SimpleLinearSchedule
from lit_sb3.utils.system import get_latest_run_id
from lit_sb3.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)

# Optuna is optional
try:
    import optuna
    from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
    from optuna.samplers import BaseSampler, RandomSampler, TPESampler
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState
    from optuna.visualization import plot_optimization_history, plot_param_importances

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore[assignment]

# Hyperparameter optimisation samplers (also needs optuna)
try:
    from lit_sb3.zoo.hyperparams_opt import HYPERPARAMS_CONVERTER, HYPERPARAMS_SAMPLER

    _HYPERPARAMS_OPT_AVAILABLE = True
except ImportError:
    _HYPERPARAMS_OPT_AVAILABLE = False


class ExperimentManager:
    """Manage a complete RL experiment: load hyperparams, create env, train and save.

    Reads algorithm hyperparameters from a YAML file (or Python module), creates
    the vectorised training and eval environments, handles callbacks and optional
    ``VecNormalize``, then runs ``model.learn()`` (which internally uses
    :class:`~lit_sb3.trainer.RLTrainer`).

    Hyperparameter optimisation (Optuna) is supported when ``optuna`` is
    installed.

    Args:
        args: Parsed argument namespace (forwarded verbatim into saved ``args.yml``).
        algo: Algorithm key, e.g. ``"ppo"``.
        env_id: Gymnasium environment ID, e.g. ``"CartPole-v1"``.
        log_folder: Root directory for logs.
        tensorboard_log: TensorBoard log directory (``""`` disabled).
        n_timesteps: Override the number of training steps (``0`` use YAML).
        eval_freq: Evaluate every n steps (``<=0`` no eval callback).
        n_eval_episodes: Episodes per evaluation.
        save_freq: Checkpoint every n steps (``<=0`` no checkpoints).
        hyperparams: Manual hyperparameter overrides.
        env_kwargs: Keyword arguments passed to the env constructor.
        eval_env_kwargs: Keyword arguments for the eval env (defaults to env_kwargs).
        trained_agent: Path to a pre-trained ``.zip`` to continue training.
        optimize_hyperparameters: Run Optuna search instead of training.
        storage: Optuna storage URL.
        study_name: Optuna study name.
        n_trials: Optuna trial budget per worker.
        max_total_trials: Hard cap across all workers.
        n_jobs: Parallel Optuna workers.
        sampler: Optuna sampler (``"tpe"``, ``"random"``, ``"auto"``).
        pruner: Optuna pruner (``"median"``, ``"halving"``, ``"none"``).
        optimization_log_path: Where to save per-trial evaluation logs.
        n_startup_trials: Trials before the Optuna sampler kicks in.
        n_evaluations: Intermediate evaluations per trial.
        truncate_last_trajectory: Truncate last HER trajectory on reload.
        uuid_str: Suffix appended to the run folder for uniqueness.
        seed: Global random seed.
        log_interval: Override log_every_n_steps (``-1`` default,
            ``<-1`` disable auto-logging).
        save_replay_buffer: Also save the off-policy replay buffer.
        verbose: Verbosity level (``0`` silent, ``1`` info).
        vec_env_type: ``"dummy"`` or ``"subproc"``.
        n_eval_envs: Parallel environments for evaluation.
        no_optim_plots: Skip Optuna visualisation plots.
        device: PyTorch device.
        config: Path to YAML / Python hyperparameter file (overrides default).
        show_progress: Display a tqdm progress bar.
        trial_id: Load hyperparameters from a specific Optuna trial.
    """

    # Subclasses can override this to use a custom VecEnv (e.g. Brax).
    default_vec_env_cls: type[VecEnv] | None = None

    def __init__(
        self,
        args: Any,
        algo: str,
        env_id: str,
        log_folder: str,
        tensorboard_log: str = "",
        n_timesteps: int = 0,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_freq: int = -1,
        hyperparams: dict[str, Any] | None = None,
        env_kwargs: dict[str, Any] | None = None,
        eval_env_kwargs: dict[str, Any] | None = None,
        trained_agent: str = "",
        optimize_hyperparameters: bool = False,
        storage: str | None = None,
        study_name: str | None = None,
        n_trials: int = 1,
        max_total_trials: int | None = None,
        n_jobs: int = 1,
        sampler: str = "tpe",
        pruner: str = "median",
        optimization_log_path: str | None = None,
        n_startup_trials: int = 0,
        n_evaluations: int | None = None,
        truncate_last_trajectory: bool = False,
        uuid_str: str = "",
        seed: int = 0,
        log_interval: int = -1,
        save_replay_buffer: bool = False,
        verbose: int = 1,
        vec_env_type: str = "dummy",
        n_eval_envs: int = 1,
        no_optim_plots: bool = False,
        device: th.device | str = "auto",
        config: str | None = None,
        show_progress: bool = False,
        trial_id: int | None = None,
    ) -> None:
        super().__init__()

        self.algo = algo
        self.env_name = EnvironmentName(env_id)
        self.custom_hyperparams = hyperparams

        # Resolve hyperparameter config file
        if (Path(__file__).parent / "hyperparams").is_dir():
            default_path = Path(__file__).parent
        else:
            default_path = Path(__file__).parent.parent

        self.config = config or str(default_path / f"hyperparams/{self.algo}.yml")
        self.env_kwargs: dict[str, Any] = env_kwargs or {}
        self.n_timesteps = n_timesteps
        self.normalize = False
        self.normalize_kwargs: dict[str, Any] = {}
        self.env_wrapper: Callable | None = None
        self._train_env: Any = None
        self.frame_stack: int | None = None
        self.seed = seed
        self.optimization_log_path = optimization_log_path

        # VecEnv class
        self.vec_env_class: type[VecEnv] = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]
        if self.default_vec_env_cls is not None:
            self.vec_env_class = self.default_vec_env_cls

        self.vec_env_wrapper: Callable | None = None
        self.vec_env_kwargs: dict[str, Any] = {}

        # Callbacks
        self.specified_callbacks: list[Any] = []
        self.callbacks: list[BaseCallback] = []

        self._eval_env_kwargs_args = eval_env_kwargs
        self.eval_env_kwargs: dict[str, Any] = eval_env_kwargs or self.env_kwargs

        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.n_envs = 1
        self.n_actions = 0
        self._hyperparams: dict[str, Any] = {}
        self.monitor_kwargs: dict[str, Any] = {}

        self.trained_agent = trained_agent
        self.continue_training = trained_agent.endswith(".zip") and Path(trained_agent).is_file()
        self.truncate_last_trajectory = truncate_last_trajectory

        self._is_atari = self._check_is_atari(env_id)

        # Optuna config
        self.optimize_hyperparameters = optimize_hyperparameters
        self.storage = storage
        self.study_name = study_name
        self.no_optim_plots = no_optim_plots
        self.trial_id = trial_id
        self.n_trials = n_trials
        self.max_total_trials = max_total_trials
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_evaluations = n_evaluations
        self.deterministic_eval = not self._is_atari

        self.device = device
        self.log_folder = log_folder
        self.tensorboard_log = None if tensorboard_log == "" else Path(tensorboard_log) / str(self.env_name)
        self.verbose = verbose
        self.args = args
        self.log_interval = log_interval
        self.save_replay_buffer = save_replay_buffer
        self.show_progress = show_progress

        self.log_path = f"{log_folder}/{self.algo}/"
        self.save_path = (
            Path(self.log_path) / f"{self.env_name}_{get_latest_run_id(self.log_path, self.env_name) + 1}{uuid_str}"
        )
        self.params_path = f"{self.save_path}/{self.env_name}"

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_point(env_id: str) -> str:
        try:
            return str(gym.envs.registry[env_id].entry_point)
        except KeyError:
            return ""

    @staticmethod
    def _check_is_atari(env_id: str) -> bool:
        try:
            return "AtariEnv" in ExperimentManager._entry_point(env_id)
        except Exception:
            return False

    @staticmethod
    def is_atari(env_id: str) -> bool:
        """Return True when env_id wraps an Atari game."""
        return ExperimentManager._check_is_atari(env_id)

    @staticmethod
    def is_minigrid(env_id: str) -> bool:
        """Return True when env_id is a MiniGrid environment."""
        try:
            return "minigrid" in ExperimentManager._entry_point(env_id)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_experiment(self) -> tuple[BaseAlgorithm, dict[str, Any]] | None:
        """Read hyperparams, create env and model.

        Returns:
            (model, saved_hyperparams) or None when running
            hyperparameter optimisation.
        """
        unprocessed_hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = self._preprocess_hyperparams(
            unprocessed_hyperparams
        )

        if self.env_kwargs:
            print(f"env_kwargs={self.env_kwargs}")
        if self.eval_env_kwargs:
            print(f"eval_env_kwargs={self.eval_env_kwargs}")

        self.create_log_folder()
        self.create_callbacks()

        n_envs = 1 if self.algo == "ars" or self.optimize_hyperparameters else self.n_envs
        env = self.create_envs(n_envs, no_log=False)

        self._hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, env)

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            env.close()
            return None
        else:
            model = ALGOS[self.algo](
                seed=self.seed,
                verbose=self.verbose,
                device=self.device,
                **self._hyperparams,
            )

        self._train_env = env
        self._save_config(saved_hyperparams)
        return model, saved_hyperparams

    def learn(self, model: BaseAlgorithm) -> None:
        """Run model.learn() with the configured parameters.

        Args:
            model: Initialised algorithm instance returned by setup_experiment.
        """
        kwargs: dict[str, Any] = {}

        # log_interval == -1  -> use algorithm default
        # log_interval == -2  -> disable auto-logging
        if self.log_interval > -1:
            kwargs["log_every_n_steps"] = self.log_interval
        elif self.log_interval < -1:
            kwargs["log_every_n_steps"] = None

        if self.callbacks:
            kwargs["callback"] = self.callbacks
        if self.tensorboard_log is not None:
            kwargs["default_root_dir"] = self.tensorboard_log

        # Special case for ARS with multiple envs
        if self.algo == "ars" and self.n_envs > 1:
            try:
                from lit_sb3.vec_env.async_eval import AsyncEval

                kwargs["async_eval"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)],
                    model.policy,
                )
            except ImportError:
                pass

        try:
            model.learn(
                self.n_timesteps,
                env=self._train_env,
                progress_bar=self.show_progress,
                **kwargs,
            )
        except KeyboardInterrupt:
            pass
        finally:
            if self.callbacks:
                self.callbacks[0].on_training_end()
            try:
                assert model.env is not None
                model.env.close()
            except EOFError:
                pass

    def save_trained_model(self, model: BaseAlgorithm) -> None:
        """Save the trained model (and optionally replay buffer + VecNormalize).

        Args:
            model: Trained algorithm instance.
        """
        print(f"Saving to {self.save_path}")
        model.save(f"{self.save_path}/{self.env_name}")

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            print("Saving replay buffer")
            model.save_replay_buffer(  # type: ignore[attr-defined]
                Path(self.save_path) / "replay_buffer.pkl"
            )

        if self.normalize:
            vec_normalize = model.get_vec_normalize_env()
            assert vec_normalize is not None
            vec_normalize.save(Path(self.params_path) / "vecnormalize.pkl")

    # ------------------------------------------------------------------
    # Hyperparameter loading
    # ------------------------------------------------------------------

    def read_hyperparameters(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load and return (unprocessed_hyperparams, saved_hyperparams)."""
        print(f"Loading hyperparameters from: {self.config}")

        if self.config.endswith((".yml", ".yaml")):
            with Path(self.config).open() as f:
                hyperparams_dict = yaml.safe_load(f)
        elif self.config.endswith(".py"):
            global_variables: dict[str, Any] = {}
            exec(Path(self.config).read_text(), global_variables)  # noqa: S102
            hyperparams_dict = global_variables["hyperparams"]
        else:
            hyperparams_dict = importlib.import_module(self.config).hyperparams

        if self.env_name.gym_id in hyperparams_dict:
            hyperparams = hyperparams_dict[self.env_name.gym_id]
        elif self._is_atari and "atari" in hyperparams_dict:
            hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_name.gym_id} in {self.config}")

        # Load from Optuna study if requested
        if _OPTUNA_AVAILABLE and self.storage and self.study_name and self.trial_id is not None:
            print("Loading from Optuna study...")
            study_hyperparams = self._load_trial(self.storage, self.study_name, self.trial_id)
            hyperparams.update(study_hyperparams)

        if self.custom_hyperparams is not None:
            hyperparams.update(self.custom_hyperparams)

        if "env_kwargs" in hyperparams and self.env_kwargs:
            hyperparams["env_kwargs"] = self.env_kwargs
            if self._eval_env_kwargs_args is None:
                self.eval_env_kwargs = hyperparams["env_kwargs"]

        saved_hyperparams = OrderedDict((key, hyperparams[key]) for key in sorted(hyperparams.keys()))

        print("Default hyperparameters for environment (ones being tuned will be overridden):")
        pprint(saved_hyperparams)

        return hyperparams, saved_hyperparams

    def _load_trial(
        self,
        storage: str,
        study_name: str,
        trial_id: int | None = None,
        convert: bool = True,
    ) -> dict[str, Any]:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for loading trial data.")
        if storage.endswith(".log"):
            optuna_storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(storage))
        else:
            optuna_storage = storage

        study = optuna.load_study(storage=optuna_storage, study_name=study_name)
        params = study.trials[trial_id].params if trial_id is not None else study.best_trial.params

        if convert and _HYPERPARAMS_OPT_AVAILABLE:
            return HYPERPARAMS_CONVERTER[self.algo](params)
        return params

    # ------------------------------------------------------------------
    # Pre-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_schedules(hyperparams: dict[str, Any]) -> dict[str, Any]:
        for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                _schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = SimpleLinearSchedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = ConstantSchedule(float(hyperparams[key]))
            else:
                raise TypeError(f"Invalid value for {key}: {hyperparams[key]}")  # noqa: TRY004
        return hyperparams

    def _preprocess_normalization(self, hyperparams: dict[str, Any]) -> dict[str, Any]:
        if "normalize" in hyperparams:
            self.normalize = hyperparams["normalize"]

            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)  # noqa: S307
                self.normalize = True

            if isinstance(self.normalize, dict):
                self.normalize_kwargs = self.normalize
                self.normalize = True

            if "gamma" in hyperparams:
                self.normalize_kwargs["gamma"] = hyperparams["gamma"]

            del hyperparams["normalize"]
        return hyperparams

    def _preprocess_hyperparams(
        self,
        hyperparams: dict[str, Any],
    ) -> tuple[dict[str, Any], Callable | None, list[BaseCallback], Callable | None]:
        self.n_envs = hyperparams.get("n_envs", 1)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        hyperparams = self._preprocess_schedules(hyperparams)

        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

        if self.n_timesteps > 0:
            if self.verbose:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])

        if self.n_evaluations is None and self.optimize_hyperparameters:
            self.n_evaluations = max(1, self.n_timesteps // int(1e5))
            print(
                f"Doing {self.n_evaluations} intermediate evaluations for pruning "
                "based on the number of timesteps. (1 evaluation every 100k timesteps)"
            )

        hyperparams = self._preprocess_normalization(hyperparams)

        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
            if kwargs_key in hyperparams and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])  # noqa: S307

        if "monitor_kwargs" in hyperparams:
            self.monitor_kwargs = hyperparams["monitor_kwargs"]
            if isinstance(self.monitor_kwargs, str):
                self.monitor_kwargs = eval(self.monitor_kwargs)  # noqa: S307
            del hyperparams["monitor_kwargs"]

        if "env_kwargs" in hyperparams:
            self.env_kwargs = hyperparams["env_kwargs"]
            del hyperparams["env_kwargs"]

        if "n_envs" in hyperparams:
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if "frame_stack" in hyperparams:
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        if "policy" in hyperparams and "." in str(hyperparams["policy"]):
            hyperparams["policy"] = get_class_by_name(hyperparams["policy"])

        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams:
            del hyperparams["env_wrapper"]

        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        if "vec_env_wrapper" in hyperparams:
            del hyperparams["vec_env_wrapper"]

        callbacks = get_callback_list(hyperparams)
        if "callback" in hyperparams:
            self.specified_callbacks = hyperparams["callback"]
            del hyperparams["callback"]

        return hyperparams, env_wrapper, callbacks, vec_env_wrapper

    def _preprocess_action_noise(
        self,
        hyperparams: dict[str, Any],
        saved_hyperparams: dict[str, Any],
        env: VecEnv,
    ) -> dict[str, Any]:
        if isinstance(env.action_space, spaces.Box):
            self.n_actions = env.action_space.shape[0]

        if hyperparams.get("noise_type") is not None:
            noise_type = hyperparams["noise_type"].strip()
            noise_std = hyperparams["noise_std"]

            assert isinstance(env.action_space, spaces.Box), (
                f"Action noise can only be used with Box action space, not {env.action_space}"
            )

            if "normal" in noise_type:
                hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            elif "ornstein-uhlenbeck" in noise_type:
                hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')

            print(f"Applying {noise_type} noise with std {noise_std}")
            del hyperparams["noise_type"]
            del hyperparams["noise_std"]

        return hyperparams

    # ------------------------------------------------------------------
    # Env creation
    # ------------------------------------------------------------------

    def create_log_folder(self) -> None:
        """Create the directory tree for this run's artefacts."""
        Path(self.params_path).mkdir(parents=True, exist_ok=True)

    def create_callbacks(self) -> None:
        """Populate self.callbacks from the current configuration."""
        if self.show_progress:
            self.callbacks.append(ProgressBarCallback())

        if self.save_freq > 0:
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.save_freq,
                    save_path=self.save_path,
                    name_prefix="rl_model",
                    verbose=1,
                )
            )

        if self.eval_freq > 0 and not self.optimize_hyperparameters:
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)
            if self.verbose > 0:
                print("Creating test environment")
            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
            eval_callback = EvalCallback(
                self.create_envs(self.n_eval_envs, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.save_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.save_path,
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval,
            )
            self.callbacks.append(eval_callback)

    def create_envs(
        self,
        n_envs: int,
        eval_env: bool = False,
        no_log: bool = False,
    ) -> VecEnv:
        """Create, wrap and return a vectorised environment.

        Args:
            n_envs: Number of parallel environments.
            eval_env: Whether this is the evaluation environment.
            no_log: Do not write a monitor log (useful for Optuna workers).

        Returns:
            Configured VecEnv.
        """
        log_dir = None if eval_env or no_log else self.save_path

        spec = gym.spec(self.env_name.gym_id)

        def make_env(**kwargs: Any) -> gym.Env:
            return spec.make(**kwargs)

        env_kwargs = self.eval_env_kwargs if eval_env else self.env_kwargs

        env = make_vec_env(
            make_env,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=self.env_wrapper,
            vec_env_cls=self.vec_env_class,  # type: ignore[arg-type]
            vec_env_kwargs=self.vec_env_kwargs,
            monitor_kwargs=self.monitor_kwargs,
        )

        if self.vec_env_wrapper is not None:
            env = self.vec_env_wrapper(env)

        env = self._maybe_normalize(env, eval_env)

        if self.frame_stack is not None:
            env = VecFrameStack(env, self.frame_stack)
            if self.verbose > 0:
                print(f"Stacking {self.frame_stack} frames")

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )
            if wrap_with_vectranspose:
                if self.verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        path_ = Path(Path(self.trained_agent).parent) / str(self.env_name)
        path_ = Path(path_) / "vecnormalize.pkl"

        if Path(path_).exists():
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            if eval_env:
                env.training = False
                env.norm_reward = False
        elif self.normalize:
            local_normalize_kwargs = self.normalize_kwargs.copy()
            if eval_env:
                local_normalize_kwargs["norm_reward"] = False
                local_normalize_kwargs["training"] = False
            if self.verbose > 0:
                if local_normalize_kwargs:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)

        return env

    # ------------------------------------------------------------------
    # Agent loading
    # ------------------------------------------------------------------

    def _load_pretrained_agent(self, hyperparams: dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        print("Loading pretrained agent")
        del hyperparams["policy"]
        if "policy_kwargs" in hyperparams:
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            verbose=self.verbose,
            device=self.device,
            **hyperparams,
        )

        replay_buffer_path = Path(Path(self.trained_agent).parent) / "replay_buffer.pkl"
        if Path(replay_buffer_path).exists():
            print("Loading replay buffer")
            assert hasattr(model, "load_replay_buffer"), "The current model does not have load_replay_buffer."
            model.load_replay_buffer(  # type: ignore[attr-defined]
                replay_buffer_path,
                truncate_last_traj=self.truncate_last_trajectory,
            )

        return model

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_config(self, saved_hyperparams: dict[str, Any]) -> None:
        with (Path(self.params_path) / "config.yml").open("w") as f:
            yaml.dump(saved_hyperparams, f)

        with (Path(self.params_path) / "args.yml").open("w") as f:
            ordered_args = OrderedDict((key, vars(self.args)[key]) for key in sorted(vars(self.args).keys()))
            yaml.dump(ordered_args, f)

        command = "python3 " + " ".join(sys.argv)
        if hasattr(sys, "orig_argv"):
            command = " ".join(sys.orig_argv)
        (Path(self.params_path) / "command.txt").write_text(command)
        print(f"Log path: {self.save_path}")

    # ------------------------------------------------------------------
    # Optuna helpers
    # ------------------------------------------------------------------

    def _create_sampler(self, sampler_method: str) -> Any:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for hyperparameter optimisation.")
        if sampler_method == "random":
            return RandomSampler(seed=self.seed)
        if sampler_method == "tpe":
            return TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed, multivariate=True)
        if sampler_method == "auto":
            try:
                import optunahub  # ty:ignore[unresolved-import]

                return optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=self.seed)
            except ImportError as exc:
                raise ImportError("optunahub is required for the 'auto' sampler.") from exc
        raise ValueError(f"Unknown sampler: {sampler_method}")

    def _create_pruner(self, pruner_method: str) -> Any:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for hyperparameter optimisation.")
        if pruner_method == "halving":
            return SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        if pruner_method == "median":
            return MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_evaluations // 3 if self.n_evaluations else 0,
            )
        if pruner_method == "none":
            return NopPruner()
        raise ValueError(f"Unknown pruner: {pruner_method}")

    def objective(self, trial: Any) -> float:
        """Optuna objective function for one trial."""
        if not _OPTUNA_AVAILABLE or not _HYPERPARAMS_OPT_AVAILABLE:
            raise ImportError("optuna and lit_sb3.zoo.hyperparams_opt are required for hyperparameter optimisation.")

        kwargs = self._hyperparams.copy()
        n_envs = 1 if self.algo == "ars" else self.n_envs

        additional_args = {
            "using_her_replay_buffer": kwargs.get("replay_buffer_class") == HerReplayBuffer,
            "her_kwargs": kwargs.get("replay_buffer_kwargs", {}),
        }

        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial, self.n_actions, n_envs, additional_args)
        kwargs.update(sampled_hyperparams)

        env = self.create_envs(n_envs, no_log=True)
        trial_verbosity = 0
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = ALGOS[self.algo](
            seed=None,
            verbose=trial_verbosity,
            device=self.device,
            **kwargs,
        )

        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)
        optuna_eval_freq = int(self.n_timesteps / (self.n_evaluations or 1))
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)

        path: str | Path | None = None
        if self.optimization_log_path is not None:
            path = Path(self.optimization_log_path) / f"trial_{trial.number!s}"

        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        learn_kwargs: dict[str, Any] = {}
        if self.algo == "ars" and self.n_envs > 1:
            try:
                from lit_sb3.vec_env.async_eval import AsyncEval

                learn_kwargs["async_eval"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)],
                    model.policy,
                )
            except ImportError:
                pass

        try:
            model.learn(
                self.n_timesteps,
                env=env,
                callback=callbacks,
                **learn_kwargs,
            )
            assert model.env is not None
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            assert model.env is not None
            model.env.close()
            eval_env.close()
            print(e)
            print("=" * 12)
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned() from e

        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        if model.env is not None:
            model.env.close()
        del eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def hyperparameters_optimization(self) -> None:
        """Run the full Optuna hyperparameter search."""
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna must be installed for hyperparameter optimisation: pip install optuna")

        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no --study-name."
                " The study name will be generated by Optuna.",
                stacklevel=2,
            )

        if self.tensorboard_log is not None:
            warnings.warn(
                "Tensorboard log is deactivated when running hyperparameter optimization",
                stacklevel=2,
            )
            self.tensorboard_log = None

        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)

        storage = self.storage
        if storage is not None and storage.endswith(".log"):
            Path(storage).parent.mkdir(parents=True, exist_ok=True)
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(storage),
            )

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            if self.max_total_trials is not None:
                counted_states = [
                    TrialState.COMPLETE,
                    TrialState.RUNNING,
                    TrialState.PRUNED,
                ]
                completed_trials = len(study.get_trials(states=counted_states))
                if completed_trials < self.max_total_trials:
                    study.optimize(
                        self.objective,
                        n_jobs=self.n_jobs,
                        callbacks=[MaxTrialsCallback(self.max_total_trials, states=counted_states)],
                    )
            else:
                study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=self.n_trials)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env_name}_{self.n_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}"
        )
        log_path = Path(self.log_folder) / self.algo / report_name

        if self.verbose:
            print(f"Writing report to {log_path}")

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        with Path(f"{log_path}.pkl").open("wb+") as f:
            pkl.dump(study, f)

        if self.no_optim_plots:
            return

        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)
            fig1.show()
            fig2.show()
        except (ValueError, ImportError, RuntimeError):
            pass

        command = "python3 " + " ".join(sys.argv)
        if hasattr(sys, "orig_argv"):
            command = " ".join(sys.orig_argv)
        (Path(self.params_path) / "command.txt").write_text(command)
        print(f"Log path: {self.save_path}")

    # ------------------------------------------------------------------
    # Optuna helpers
    # ------------------------------------------------------------------

    def _create_sampler(self, sampler_method: str) -> Any:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for hyperparameter optimisation.")
        if sampler_method == "random":
            return RandomSampler(seed=self.seed)
        if sampler_method == "tpe":
            return TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed, multivariate=True)
        if sampler_method == "auto":
            try:
                import optunahub  # ty:ignore[unresolved-import]

                return optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=self.seed)
            except ImportError as exc:
                raise ImportError("optunahub is required for the 'auto' sampler.") from exc
        raise ValueError(f"Unknown sampler: {sampler_method}")

    def _create_pruner(self, pruner_method: str) -> Any:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for hyperparameter optimisation.")
        if pruner_method == "halving":
            return SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        if pruner_method == "median":
            return MedianPruner(
                n_startup_trials=self.n_startup_trials,
                n_warmup_steps=self.n_evaluations // 3 if self.n_evaluations else 0,
            )
        if pruner_method == "none":
            return NopPruner()
        raise ValueError(f"Unknown pruner: {pruner_method}")

    def objective(self, trial: Any) -> float:  # noqa: F811
        """Optuna objective function for one trial."""
        if not _OPTUNA_AVAILABLE or not _HYPERPARAMS_OPT_AVAILABLE:
            raise ImportError("optuna and lit_sb3.zoo.hyperparams_opt are required for hyperparameter optimisation.")

        kwargs = self._hyperparams.copy()
        n_envs = 1 if self.algo == "ars" else self.n_envs

        additional_args = {
            "using_her_replay_buffer": kwargs.get("replay_buffer_class") == HerReplayBuffer,
            "her_kwargs": kwargs.get("replay_buffer_kwargs", {}),
        }

        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial, self.n_actions, n_envs, additional_args)
        kwargs.update(sampled_hyperparams)

        env = self.create_envs(n_envs, no_log=True)
        trial_verbosity = 0
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = ALGOS[self.algo](
            seed=None,
            verbose=trial_verbosity,
            device=self.device,
            **kwargs,
        )

        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)
        optuna_eval_freq = int(self.n_timesteps / (self.n_evaluations or 1))
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)

        path: str | Path | None = None
        if self.optimization_log_path is not None:
            path = Path(self.optimization_log_path) / f"trial_{trial.number!s}"

        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        learn_kwargs: dict[str, Any] = {}
        if self.algo == "ars" and self.n_envs > 1:
            try:
                from lit_sb3.vec_env.async_eval import AsyncEval

                learn_kwargs["async_eval"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)],
                    model.policy,
                )
            except ImportError:
                pass

        try:
            model.learn(
                self.n_timesteps,
                env=env,
                callback=callbacks,
                **learn_kwargs,
            )
            assert model.env is not None
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            assert model.env is not None
            model.env.close()
            eval_env.close()
            print(e)
            print("=" * 12)
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned() from e

        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        if model.env is not None:
            model.env.close()
        del eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def hyperparameters_optimization(self) -> None:  # noqa: F811
        """Run the full Optuna hyperparameter search."""
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna must be installed for hyperparameter optimisation: pip install optuna")

        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no --study-name."
                " The study name will be generated by Optuna.",
                stacklevel=2,
            )

        if self.tensorboard_log is not None:
            warnings.warn(
                "Tensorboard log is deactivated when running hyperparameter optimization",
                stacklevel=2,
            )
            self.tensorboard_log = None

        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)

        storage = self.storage
        if storage is not None and storage.endswith(".log"):
            Path(storage).parent.mkdir(parents=True, exist_ok=True)
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(storage),
            )

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            if self.max_total_trials is not None:
                counted_states = [
                    TrialState.COMPLETE,
                    TrialState.RUNNING,
                    TrialState.PRUNED,
                ]
                completed_trials = len(study.get_trials(states=counted_states))
                if completed_trials < self.max_total_trials:
                    study.optimize(
                        self.objective,
                        n_jobs=self.n_jobs,
                        callbacks=[MaxTrialsCallback(self.max_total_trials, states=counted_states)],
                    )
            else:
                study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=self.n_trials)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env_name}_{self.n_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}"
        )
        log_path = Path(self.log_folder) / self.algo / report_name

        if self.verbose:
            print(f"Writing report to {log_path}")

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        with Path(f"{log_path}.pkl").open("wb+") as f:
            pkl.dump(study, f)

        if self.no_optim_plots:
            return

        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)
            fig1.show()
            fig2.show()
        except (ValueError, ImportError, RuntimeError):
            pass
        with Path(f"{log_path}.pkl").open("wb+") as f:
            pkl.dump(study, f)

        if self.no_optim_plots:
            return

        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)
            fig1.show()
            fig2.show()
        except (ValueError, ImportError, RuntimeError):
            pass
