"""Trained model path and hyperparameter loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lit_sb3.huggingface.naming_schemes import EnvironmentName, ModelName
from lit_sb3.utils.registry import ALGOS
from lit_sb3.utils.system import get_latest_run_id


def get_saved_hyperparams(
    stats_path: str | Path,
    norm_reward: bool = False,
    test_mode: bool = False,
) -> tuple[dict[str, Any], str | Path | None]:
    """Load hyperparameters saved by :class:`~lit_sb3.core.exp_manager.ExperimentManager`.

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
        repo_name = model_id.split("/")[-1]
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
        exp_id = get_latest_run_id(str(Path(folder) / algo), str(env_name))
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

    model_path = str(Path(log_path) / f"{env_name}.zip")
    if not Path(model_path).is_file():
        model_paths = [str(m) for m in Path(log_path).glob("*.zip") if "best_model" not in m.name]
        if model_paths:
            model_path = model_paths[0]
        else:
            raise ValueError(f"No model found for {name_prefix} on {env_name}, path: {log_path}")

    return name_prefix, model_path, log_path
