"""Download models from the HuggingFace Hub.

Integrates:

- ``load_from_hub``: thin HF download function, adapted from
  https://github.com/huggingface/huggingface_sb3/blob/main/huggingface_sb3/load_from_hub.py
- ``download_from_hub``: RL-Zoo3-format downloader, adapted from
  https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/load_from_hub.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import zipfile

from lit_sb3.huggingface.naming_schemes import EnvironmentName, ModelName, ModelRepoId


def load_from_hub(repo_id: str, filename: str) -> str:
    """Download a single file from the HuggingFace Hub.

    Adapted from
    https://github.com/huggingface/huggingface_sb3/blob/main/huggingface_sb3/load_from_hub.py

    :param repo_id: Repository id from the HuggingFace Hub
        (e.g. ``"sb3/ppo-CartPole-v1"``).
    :param filename: Name of the file to download from the repository.
    :return: Local path to the downloaded file.
    :raises ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        ) from exc
    downloaded_model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        library_name="huggingface-sb3",
        library_version="2.1",
    )
    return downloaded_model_file


def download_from_hub(
    algo: str,
    env_name: EnvironmentName,
    exp_id: int,
    folder: str,
    organization: str,
    repo_name: str | None = None,
    force: bool = False,
) -> None:
    """Download a model from the HuggingFace Hub and save it in RL-Zoo3 format.

    Adapted from
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/load_from_hub.py

    Default repo is ``{organization}/{algo}-{env_id}`` where
    ``repo_name = {algo}-{env_id}``.

    :param algo: Algorithm name (e.g. ``"ppo"``).
    :param env_name: Environment name.
    :param exp_id: Experiment ID (0 = auto-increment to latest + 1).
    :param folder: Root log folder.
    :param organization: HuggingFace organisation name.
    :param repo_name: Override the default ``{algo}-{env_id}`` repository name.
    :param force: Allow overwriting if the target folder already exists.
    :raises ImportError: If ``huggingface_hub`` is not installed.
    :raises ValueError: If the target folder already exists and *force* is ``False``.
    """
    # local import to keep the module importable without huggingface_hub installed
    from lit_sb3.utils.system import get_latest_run_id

    model_name = ModelName(algo, env_name)
    if repo_name is None:
        repo_name = str(model_name)

    repo_id = ModelRepoId(organization, repo_name)
    print(f"Downloading from https://huggingface.co/{repo_id}")

    checkpoint = load_from_hub(str(repo_id), model_name.filename)
    config_path = load_from_hub(str(repo_id), "config.yml")

    # VecNormalize stats are optional
    vec_normalize_stats: str | None = None
    try:
        vec_normalize_stats = load_from_hub(str(repo_id), "vec_normalize.pkl")
    except Exception:
        print("No normalization file found in the repository.")

    saved_args = load_from_hub(str(repo_id), "args.yml")
    env_kwargs_file = load_from_hub(str(repo_id), "env_kwargs.yml")
    train_eval_metrics = load_from_hub(str(repo_id), "train_eval_metrics.zip")

    if exp_id == 0:
        exp_id = get_latest_run_id(Path(folder) / algo, env_name) + 1

    log_path = Path(folder) / algo / f"{env_name}_{exp_id}" if exp_id > 0 else Path(folder) / algo

    log_folder = Path(log_path)
    if log_folder.is_dir():
        if force:
            print(f"The folder {log_path} already exists, overwriting.")
            shutil.rmtree(log_path)
        else:
            raise ValueError(
                f"The folder {log_path} already exists, use --force to overwrite it, "
                "or choose '--exp-id 0' to auto-create a new folder."
            )

    print(f"Saving to {log_path}")
    Path(log_path).mkdir(parents=True, exist_ok=True)
    config_folder = Path(log_path) / str(env_name)
    Path(config_folder).mkdir(parents=True, exist_ok=True)

    shutil.copy(checkpoint, Path(log_path) / f"{env_name}.zip")
    shutil.copy(saved_args, Path(config_folder) / "args.yml")
    shutil.copy(config_path, Path(config_folder) / "config.yml")
    shutil.copy(env_kwargs_file, Path(config_folder) / "env_kwargs.yml")
    if vec_normalize_stats is not None:
        shutil.copy(vec_normalize_stats, Path(config_folder) / "vecnormalize.pkl")

    with zipfile.ZipFile(train_eval_metrics, "r") as zip_ref:
        zip_ref.extractall(log_path)
