"""Zoo utilities: ALGOS registry, schedule helpers, env/model utilities.

Adapted from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/utils.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
import importlib
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from lit_sb3.algorithms.a2c.a2c import A2C
from lit_sb3.algorithms.ars.ars import ARS
from lit_sb3.algorithms.base import BaseAlgorithm
from lit_sb3.algorithms.crossq.crossq import CrossQ
from lit_sb3.algorithms.ddpg.ddpg import DDPG
from lit_sb3.algorithms.dqn.dqn import DQN
from lit_sb3.algorithms.ppo.ppo import PPO
from lit_sb3.algorithms.ppo_recurrent.ppo_recurrent import RecurrentPPO
from lit_sb3.algorithms.qrdqn.qrdqn import QRDQN
from lit_sb3.algorithms.sac.sac import SAC
from lit_sb3.algorithms.td3.td3 import TD3
from lit_sb3.algorithms.tqc.tqc import TQC
from lit_sb3.algorithms.trpo.trpo import TRPO
from lit_sb3.callbacks import BaseCallback
from lit_sb3.core.preprocessing import is_image_space, is_image_space_channels_first
from lit_sb3.huggingface.naming_schemes import EnvironmentName, ModelName, ModelRepoId
from lit_sb3.utils.env_util import make_vec_env
from lit_sb3.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
)
from lit_sb3.wrappers.monitor import Monitor  # noqa: F401 (re-exported for user convenience)

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGOS: dict[str, type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ars": ARS,
    "crossq": CrossQ,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "ppo_lstm": RecurrentPPO,
    "qrdqn": QRDQN,
    "sac": SAC,
    "td3": TD3,
    "tqc": TQC,
    "trpo": TRPO,
}


# ---------------------------------------------------------------------------

# Schedule helpers
# ---------------------------------------------------------------------------


class SimpleLinearSchedule:
    """Linear schedule from *initial_value* down to 0 as training progresses.

    Args:
        initial_value: Starting value (must be positive).
    """

    def __init__(self, initial_value: float | str) -> None:
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value

    def __repr__(self) -> str:
        return f"SimpleLinearSchedule(initial_value={self.initial_value})"


def linear_schedule(initial_value: float | str) -> SimpleLinearSchedule:
    """Return a :class:`SimpleLinearSchedule` for *initial_value*."""
    return SimpleLinearSchedule(initial_value)


# ---------------------------------------------------------------------------
# Wrapper / callback discovery helpers
# ---------------------------------------------------------------------------


def _get_module_name(dotted: str) -> str:
    return ".".join(dotted.split(".")[:-1])


def _get_class_name(dotted: str) -> str:
    return dotted.split(".")[-1]


def get_class_by_name(name: str) -> type:
    """Import and return a class given a dotted name string.

    Example::

        get_class_by_name("lit_sb3.callbacks.CheckpointCallback")
    """
    module = importlib.import_module(_get_module_name(name))
    return getattr(module, _get_class_name(name))


def get_wrapper_class(
    hyperparams: dict[str, Any],
    key: str = "env_wrapper",
) -> Callable[[gym.Env], gym.Env] | None:
    """Return a callable that applies one or more gym wrappers.

    The ``key`` entry in *hyperparams* may be a single dotted class name or a
    list.  Each entry may optionally be a ``{ClassName: kwargs_dict}`` mapping
    to pass keyword arguments to the wrapper constructor.

    Args:
        hyperparams: Hyperparameter dict that may contain ``key``.
        key: The hyperparams key to inspect (``"env_wrapper"`` or
            ``"vec_env_wrapper"``).

    Returns:
        A callable ``wrap_env(env) -> env`` or ``None`` if the key is absent.
    """
    if key not in hyperparams:
        return None

    wrapper_name = hyperparams.get(key)
    if wrapper_name is None:
        return None

    wrapper_names: list[Any] = wrapper_name if isinstance(wrapper_name, list) else [wrapper_name]

    wrapper_classes: list[type] = []
    wrapper_kwargs_list: list[dict[str, Any]] = []

    for entry in wrapper_names:
        if isinstance(entry, dict):
            assert len(entry) == 1, f"Error in YAML near {entry}: expected single-key dict."
            class_name = next(iter(entry.keys()))
            kwargs: dict[str, Any] = entry[class_name]
        else:
            class_name = entry
            kwargs = {}

        if isinstance(class_name, str):
            module = importlib.import_module(_get_module_name(class_name))
            wrapper_class = getattr(module, _get_class_name(class_name))
        elif isinstance(class_name, type):
            wrapper_class = class_name
        else:
            raise TypeError(f"Unexpected value {class_name!r} for {key}; must be a str or class.")

        wrapper_classes.append(wrapper_class)
        wrapper_kwargs_list.append(kwargs)

    def wrap_env(env: gym.Env) -> gym.Env:
        for cls, kw in zip(wrapper_classes, wrapper_kwargs_list, strict=True):
            env = cls(env, **kw)
        return env

    return wrap_env


def get_callback_list(hyperparams: dict[str, Any]) -> list[BaseCallback]:
    r"""Return a list of instantiated callbacks from *hyperparams[\"callback\"]*.

    Each entry can be:
    * a dotted class name string,
    * ``{ClassName: kwargs_dict}``, or
    * an already-instantiated :class:`~lit_sb3.callbacks.BaseCallback`.
    """
    callbacks: list[BaseCallback] = []
    if "callback" not in hyperparams:
        return callbacks

    callback_name = hyperparams.get("callback")
    if callback_name is None:
        return callbacks

    callback_names: list[Any] = callback_name if isinstance(callback_name, list) else [callback_name]

    for entry in callback_names:
        if isinstance(entry, dict):
            assert len(entry) == 1, f"Error in YAML near {entry}: expected single-key dict."
            name = next(iter(entry.keys()))
            kwargs = entry[name]
        else:
            name = entry
            kwargs = {}

        if isinstance(name, BaseCallback):
            callbacks.append(name)
        else:
            assert isinstance(name, str), f"Expected str callback name, got {type(name)}"
            callback_class = get_class_by_name(name)
            callbacks.append(callback_class(**kwargs))

    return callbacks


# ---------------------------------------------------------------------------
# Environment creation helpers
# ---------------------------------------------------------------------------


def create_test_env(
    env_id: str,
    n_envs: int = 1,
    stats_path: str | Path | None = None,
    seed: int = 0,
    log_dir: str | Path | None = None,
    should_render: bool = True,
    hyperparams: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    vec_env_cls: type[VecEnv] | None = None,
    vec_env_kwargs: dict[str, Any] | None = None,
) -> VecEnv:
    """Create a vectorised environment for evaluating a trained agent.

    Args:
        env_id: Gymnasium environment ID.
        n_envs: Number of parallel environments.
        stats_path: Folder containing saved ``VecNormalize`` stats.
        seed: Random seed.
        log_dir: Optional directory for episode reward logging.
        should_render: Add ``render_mode="human"`` when ``True``.
        hyperparams: Extra hyperparams (e.g. ``frame_stack``, ``normalize``).
        env_kwargs: Keyword arguments passed to the env constructor.
        vec_env_cls: VecEnv class to use (defaults to ``DummyVecEnv``).
        vec_env_kwargs: Keyword arguments for the VecEnv constructor.

    Returns:
        Wrapped :class:`~lit_sb3.vec_env.VecEnv`.
    """
    assert hyperparams is not None
    env_wrapper = get_wrapper_class(hyperparams)

    # Take a copy so we do not mutate the caller's dict
    hyperparams = {} if hyperparams is None else hyperparams.copy()

    if "env_wrapper" in hyperparams:
        del hyperparams["env_wrapper"]

    if vec_env_cls is None:
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    env_kwargs = deepcopy(env_kwargs or {})
    if "render_mode" not in env_kwargs and should_render:
        env_kwargs["render_mode"] = "human"

    spec = gym.spec(env_id)

    def make_env(**kwargs: Any) -> gym.Env:
        return spec.make(**kwargs)

    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        wrapper_class=env_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,  # type: ignore[arg-type]
        vec_env_kwargs=vec_env_kwargs,
    )

    if "vec_env_wrapper" in hyperparams:
        vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
        assert vec_env_wrapper is not None
        env = vec_env_wrapper(env)  # type: ignore[arg-type]
        assert isinstance(env, VecEnv), "vec_env_wrapper must return a VecEnv"
        del hyperparams["vec_env_wrapper"]

    # Load saved normalisation stats if available
    if stats_path is not None:
        if hyperparams.get("normalize"):
            print("Loading running average")
            print(f"with params: {hyperparams.get('normalize_kwargs', {})}")
            path_ = Path(stats_path) / "vecnormalize.pkl"
            if Path(path_).exists():
                env = VecNormalize.load(path_, env)
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)

    return env


# ---------------------------------------------------------------------------
# Log / model path helpers
# ---------------------------------------------------------------------------


def get_latest_run_id(log_path: str | Path, env_name: str) -> int:
    """Return the highest run-ID found in *log_path* for *env_name*.

    Args:
        log_path: Root directory that contains run folders.
        env_name: Environment name used as folder prefix.

    Returns:
        Latest integer run ID, or 0 if none found.
    """
    max_run_id = 0
    for path in (str(p) for p in Path(log_path).glob(f"{env_name}_[0-9]*")):
        run_id = path.split("_")[-1]
        path_without_run_id = path[: -len(run_id) - 1]
        if path_without_run_id.endswith(str(env_name)) and run_id.isdigit() and int(run_id) > max_run_id:
            max_run_id = int(run_id)
    return max_run_id


def get_saved_hyperparams(
    stats_path: str | Path,
    norm_reward: bool = False,
    test_mode: bool = False,
) -> tuple[dict[str, Any], str | Path | None]:
    """Load hyperparameters saved by :class:`~lit_sb3.zoo.exp_manager.ExperimentManager`.

    Args:
        stats_path: Path to the folder containing ``config.yml``.
        norm_reward: Whether to keep reward normalisation active.
        test_mode: Disable reward normalisation and training flag.

    Returns:
        Tuple of ``(hyperparams_dict, stats_path_or_None)``.
    """
    hyperparams: dict[str, Any] = {}
    if not Path(stats_path).is_dir():
        return hyperparams, None

    config_file = Path(stats_path) / "config.yml"
    if Path(config_file).is_file():
        with Path(config_file).open() as f:
            hyperparams = yaml.safe_load(f)
        hyperparams["normalize"] = hyperparams.get("normalize", False)
    else:
        obs_rms_path = Path(stats_path) / "obs_rms.pkl"
        hyperparams["normalize"] = Path(obs_rms_path).is_file()

    if hyperparams["normalize"]:
        if isinstance(hyperparams["normalize"], str):
            normalize_kwargs = eval(hyperparams["normalize"])  # noqa: S307
            if test_mode:
                normalize_kwargs["norm_reward"] = norm_reward
        elif isinstance(hyperparams["normalize"], dict):
            normalize_kwargs = hyperparams["normalize"]
        else:
            normalize_kwargs = {
                "norm_obs": hyperparams["normalize"],
                "norm_reward": norm_reward,
            }
        hyperparams["normalize_kwargs"] = normalize_kwargs

    return hyperparams, stats_path


def get_hf_trained_models(organization: str = "sb3") -> dict[str, tuple[str, str]]:
    """Fetch trained models from a HuggingFace organisation.

    Returns a dict mapping model names to ``(algo, env_id)`` tuples,
    mirroring :func:`get_trained_models`.

    Requires ``huggingface_hub`` to be installed.

    :param organization: HuggingFace organisation to search.
    :return: Dict mapping model names to ``(algo, env_id)`` tuples.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to fetch models from the hub. Install it with: pip install huggingface_hub"
        ) from e

    api = HfApi()
    models = api.list_models(author=organization, cardData=True)

    trained_models: dict[str, tuple[str, str]] = {}
    for model_info in models:
        model_id = getattr(model_info, "modelId", None) or getattr(model_info, "id", None) or ""
        # repo name format: {algo}-{env_id}
        repo_name = model_id.split("/")[-1]
        # split on "-" but env_id may contain "-" too, so match against ALGOS
        for algo in ALGOS:
            prefix = f"{algo}-"
            if repo_name.startswith(prefix):
                env_id = repo_name[len(prefix) :]
                env_name = EnvironmentName(env_id)
                model_name = ModelName(algo, env_name)
                trained_models[str(model_name)] = (algo, env_id)
                break

    return trained_models


def get_trained_models(log_folder: str) -> dict[str, tuple[str, str]]:
    """Scan *log_folder* for trained agents saved by the ExperimentManager.

    Args:
        log_folder: Root log directory (same as ``--log-folder`` in training).

    Returns:
        Dict mapping ``ModelName`` strings to ``(algo, env_id)`` tuples.
    """
    trained_models: dict[str, tuple[str, str]] = {}
    for algo in list(Path(log_folder).iterdir()):
        algo_str = str(algo)
        if not (Path(log_folder) / algo_str).is_dir():
            continue
        for env_id in (x.name for x in (Path(log_folder) / algo_str).iterdir()):
            for _model_path in (Path(log_folder) / algo_str / env_id).glob("*.zip"):
                try:
                    env_name = EnvironmentName(env_id)
                    model_name = ModelName(algo_str, env_name)
                    trained_models[str(model_name)] = (algo_str, env_id)
                    break
                except Exception:  # noqa: S112
                    continue
    return trained_models


class StoreDict(argparse.Action):
    """Custom ``argparse`` action for storing key-value pairs in a dict.

    Enables ``key1:val1 key2:val2`` syntax on the CLI.
    Numeric strings are auto-converted to ``int`` or ``float``.
    Comma-separated values become lists.
    """

    def __init__(self, option_strings: list[str], dest: str, nargs: str = "+", **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | list[str] | None,
        option_string: str | None = None,
    ) -> None:  # ty:ignore[invalid-method-override]
        if values is None:
            return
        args: dict[str, Any] = {}
        for item in values:
            k, v = item.split(":")
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    if isinstance(v, str) and "," in v:
                        v = v.split(",")
            args[k] = v
        setattr(namespace, self.dest, args)


def get_model_path(
    exp_id: int,
    folder: str,
    algo: str,
    env_name: EnvironmentName,
    load_best: bool = False,
    load_checkpoint: str | None = None,
    load_last_checkpoint: bool = False,
) -> tuple[str, str, str]:
    """Retrieve the path to a trained model, along with its log directory.

    Args:
        exp_id: Experiment ID (0 → use the latest available run).
        folder: Root folder where run directories are stored (e.g. ``logs/``).
        algo: Algorithm name, e.g. ``"ppo"``.
        env_name: :class:`EnvironmentName` wrapper for the target environment.
        load_best: When ``True``, load ``best_model.zip`` instead of the final
            checkpoint.
        load_checkpoint: If non-``None``, load the model saved at this specific
            step count (e.g. ``"10000"``).
        load_last_checkpoint: When ``True``, load the numerically largest
            step-count checkpoint rather than the final model.

    Returns:
        ``(name_prefix, model_path, log_path)`` where *name_prefix* is a
        human-readable identifier, *model_path* points to the ``.zip`` file,
        and *log_path* is the run directory.
    """
    if exp_id == 0:
        exp_id = get_latest_run_id(str(Path(folder) / algo), env_name)
        print(f"Loading latest experiment, id={exp_id}")

    log_path = str(Path(folder) / algo / f"{env_name}_{exp_id}") if exp_id > 0 else str(Path(folder) / algo)

    assert Path(log_path).is_dir(), (
        f"No logs found for {log_path}. Use '--folder' to specify the correct log directory."
    )

    name_prefix = f"{algo}-{env_name}"

    if load_best:
        model_path = str(Path(log_path) / "best_model.zip")
        assert Path(model_path).is_file(), f"No best model found at {model_path}"
        return name_prefix, model_path, log_path

    if load_checkpoint is not None:
        model_path = str(Path(log_path) / f"rl_model_{load_checkpoint}_steps.zip")
        assert Path(model_path).is_file(), f"No checkpoint found at {model_path}"
        return name_prefix, model_path, log_path

    if load_last_checkpoint:
        checkpoints = [str(p) for p in Path(log_path).glob("rl_model_*_steps.zip")]
        if not checkpoints:
            raise ValueError(f"No checkpoint found for {name_prefix} on {env_name}, path: {log_path}")

        def _steps(ckpt: str) -> int:
            filename = Path(ckpt).name
            return int(filename.replace("rl_model_", "").replace("_steps.zip", ""))

        checkpoints = sorted(checkpoints, key=_steps)
        return name_prefix, checkpoints[-1], log_path

    # Load default final model
    model_path = str(Path(log_path) / f"{env_name}.zip")
    if not Path(model_path).is_file():
        model_paths = [str(m) for m in Path(log_path).glob("*.zip") if "best_model" not in m.name]
        if model_paths:
            model_path = model_paths[0]
        else:
            raise ValueError(f"No model found for {name_prefix} on {env_name}, path: {log_path}")

    return name_prefix, model_path, log_path
