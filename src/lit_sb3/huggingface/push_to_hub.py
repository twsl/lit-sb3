"""Push trained models to the HuggingFace Hub.

Integrates helpers from:

- https://github.com/huggingface/huggingface_sb3/blob/main/huggingface_sb3/push_to_hub.py
  (``_evaluate_agent``, ``_generate_replay``, ``_generate_metadata``, ``push_to_hub``)
- https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/push_to_hub.py
  (``generate_model_card``, ``save_model_card``, ``package_to_hub``)

All stable_baselines3 imports replaced with lit_sb3 equivalents.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from pprint import pformat
import shutil
import tempfile
from typing import Any
import zipfile

import numpy as np
import yaml

from lit_sb3.__about__ import __version__
from lit_sb3.core.evaluation import evaluate_policy
from lit_sb3.huggingface.naming_schemes import EnvironmentName, ModelName, ModelRepoId
from lit_sb3.vec_env import DummyVecEnv, VecEnv, VecVideoRecorder, unwrap_vec_normalize

try:
    from huggingface_hub import HfApi, ModelCard, ModelCardData
    from huggingface_hub.repocard import metadata_eval_result, metadata_save

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Internal helpers (adapted from huggingface_sb3.push_to_hub)
# ---------------------------------------------------------------------------


def _generate_metadata(
    model_name: str,
    env_id: str,
    mean_reward: float,
    std_reward: float,
) -> Any:
    """Build :class:`huggingface_hub.ModelCardData` with eval metrics tags.

    Adapted from ``huggingface_sb3.push_to_hub._generate_metadata``.

    Args:
        model_name: Algorithm class name (e.g. ``"PPO"``).
        env_id: Gym environment id.
        mean_reward: Mean evaluation reward.
        std_reward: Standard deviation of evaluation reward.

    Returns:
        ``ModelCardData`` instance, or plain ``dict`` if huggingface_hub
        is not installed.
    """
    if not _HF_AVAILABLE:
        return {"mean_reward": mean_reward, "std_reward": std_reward}

    metadata: dict[str, Any] = {}
    metadata["library_name"] = "stable-baselines3"
    metadata["tags"] = [
        env_id,
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "stable-baselines3",
    ]
    eval_result = metadata_eval_result(
        model_pretty_name=model_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_id,
        dataset_id=env_id,
    )
    return ModelCardData(**metadata, **eval_result)


def _evaluate_agent(
    model: Any,
    eval_env: VecEnv,
    n_eval_episodes: int,
    is_deterministic: bool,
    local_path: Path,
) -> tuple[float, float]:
    """Evaluate the agent with :func:`evaluate_policy` and write ``results.json``.

    Adapted from ``huggingface_sb3.push_to_hub._evaluate_agent``.

    Args:
        model: Trained model.
        eval_env: Evaluation environment.
        n_eval_episodes: Number of evaluation episodes.
        is_deterministic: Use deterministic actions.
        local_path: Directory for writing ``results.json``.

    Returns:
        ``(mean_reward, std_reward)`` tuple.
    """
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes, is_deterministic)

    evaluate_data: dict[str, Any] = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "is_deterministic": is_deterministic,
        "n_eval_episodes": n_eval_episodes,
        "eval_datetime": datetime.datetime.now().isoformat(),
    }
    with Path(local_path / "results.json").open("w") as outfile:
        json.dump(evaluate_data, outfile)

    return float(mean_reward), float(std_reward)  # ty:ignore[invalid-argument-type]


def _generate_replay(
    model: Any,
    eval_env: VecEnv,
    video_length: int,
    is_deterministic: bool,
    local_path: Path,
) -> None:
    """Record a replay video and save it as ``replay.mp4`` in *local_path*.

    Adapted from ``huggingface_sb3.push_to_hub._generate_replay``.

    Args:
        model: Trained model.
        eval_env: Evaluation environment.
        video_length: Number of timesteps to record.
        is_deterministic: Use deterministic actions.
        local_path: Output directory for ``replay.mp4``.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        env = VecVideoRecorder(
            eval_env,
            tmpdirname,
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix="",
        )

        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        try:
            for _ in range(video_length):
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=is_deterministic,
                )
                obs, _, episode_starts, _ = env.step(action)

            env.close()
            inp = env.video_recorder.path
            out = str(local_path / "replay.mp4")
            os.system(f"ffmpeg -y -i {inp} -vcodec h264 {out}")  # noqa: S605

        except KeyboardInterrupt:
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(str(exc), style="bold red")
            console.print("Unable to generate the replay. The push process continues.", style="bold red")


# ---------------------------------------------------------------------------
# Public helpers (RL-Zoo3 style, adapted from rl_zoo3.push_to_hub)
# ---------------------------------------------------------------------------


def generate_model_card(
    algo_name: str,
    algo_class_name: str,
    organization: str,
    env_id: str,
    mean_reward: float,
    std_reward: float,
    hyperparams: dict[str, Any],
    env_kwargs: dict[str, Any],
) -> tuple[str, Any]:
    """Generate a model card for the HuggingFace Hub (RL-Zoo3 style).

    Adapted from
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/push_to_hub.py

    Args:
        algo_name: Algorithm alias (e.g. ``"ppo"``).
        algo_class_name: Algorithm class name (e.g. ``"PPO"``).
        organization: HuggingFace organisation.
        env_id: Gym environment ID.
        mean_reward: Mean reward of the evaluated agent.
        std_reward: Standard deviation of the mean reward.
        hyperparams: Hyperparameters used for training.
        env_kwargs: Additional env keyword arguments.

    Returns:
        Tuple of ``(model_card_text, metadata)``.
    """
    metadata = _generate_metadata(algo_class_name, env_id, mean_reward, std_reward)

    model_card = f"""
# **{algo_class_name}** Agent playing **{env_id}**

This is a trained model of a **{algo_class_name}** agent playing **{env_id}**
using the [lit-sb3 library](https://github.com/twsl/lit-sb3)
and the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

## Usage (with lit-sb3 RL Zoo)

RL Zoo: https://github.com/DLR-RM/rl-baselines3-zoo<br/>
lit-sb3: https://github.com/twsl/lit-sb3

```bash
# Download model and save it into the logs/ folder
lit-sb3 load_from_hub --algo {algo_name} --env {env_id} -orga {organization} -f logs/
lit-sb3 enjoy --algo {algo_name} --env {env_id} -f logs/
```

## Training (with the RL Zoo)
```
lit-sb3 train --algo {algo_name} --env {env_id} -f logs/
# Upload the model and generate video (when possible)
lit-sb3 push_to_hub --algo {algo_name} --env {env_id} -f logs/ -orga {organization}
```

## Hyperparameters
```python
{pformat(hyperparams)}
```
"""
    if env_kwargs:
        model_card += f"""
## Environment Arguments
```python
{pformat(env_kwargs)}
```
"""
    return model_card, metadata


def save_model_card(
    repo_dir: Path,
    generated_model_card: str,
    metadata: Any,
) -> None:
    """Write ``README.md`` (model card) to *repo_dir*.

    Args:
        repo_dir: Repository directory.
        generated_model_card: Model card text.
        metadata: Metadata to embed in README front-matter.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    if not _HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to save the model card. Install with: pip install huggingface_hub"
        )
    readme_path = repo_dir / "README.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(generated_model_card)
    metadata_save(readme_path, metadata)


# ---------------------------------------------------------------------------
# Push functions
# ---------------------------------------------------------------------------


def package_to_hub(
    model: Any,
    model_name: ModelName,
    algo_name: str,
    algo_class_name: str,
    log_path: Path,
    hyperparams: dict[str, Any],
    env_kwargs: dict[str, Any],
    env_name: EnvironmentName,
    eval_env: Any,
    repo_id: Any,
    commit_message: str,
    is_deterministic: bool = True,
    n_eval_episodes: int = 10,
    token: str | None = None,
    local_repo_path: str = "hub",
    video_length: int = 1000,
    generate_video: bool = False,
) -> str:
    """Evaluate, generate a video and upload a model to the HuggingFace Hub.

    Adapted from
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/push_to_hub.py

    This function:

    1. Clones (or creates) the HuggingFace repository locally.
    2. Saves the model and optional ``VecNormalize`` stats.
    3. Copies training config / metric files.
    4. Evaluates the agent and writes ``results.json``.
    5. Optionally records a ``replay.mp4``.
    6. Generates and saves the model card.
    7. Uploads everything to the Hub.

    Args:
        model: Trained model.
        model_name: :class:`~lit_sb3.huggingface.naming_schemes.ModelName`.
        algo_name: Algorithm alias (e.g. ``"ppo"``).
        algo_class_name: Algorithm class name (e.g. ``"PPO"``).
        log_path: Path to the training output folder.
        hyperparams: Hyperparameters used for training.
        env_kwargs: Additional env keyword arguments.
        env_name: Environment name.
        eval_env: Environment used to evaluate the agent.
        repo_id: HuggingFace repository id (``"{org}/{name}"``).
        commit_message: Commit message for the Hub upload.
        is_deterministic: Use deterministic actions.
        n_eval_episodes: Number of evaluation episodes.
        token: HuggingFace authentication token.
        local_repo_path: Local base directory for the cloned repo.
        video_length: Video length in timesteps.
        generate_video: Whether to record a replay video.

    Returns:
        Repository URL.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    if not _HF_AVAILABLE:
        raise ImportError("huggingface_hub is required to push to the Hub. Install with: pip install huggingface_hub")

    console.print(
        "This function will save, evaluate, generate a video of your agent, "
        "create a model card and push everything to the hub. "
        "It might take up to some minutes if video generation is activated. "
        "This is a work in progress: if you encounter a bug, please open an issue.",
        style="bold blue",
    )

    _organization, repo_name = str(repo_id).split("/")

    api = HfApi()
    repo_url = api.create_repo(
        token=token,
        repo_id=str(repo_id),
        private=False,
        exist_ok=True,
    )

    repo_local_path = Path(local_repo_path) / repo_name
    api.snapshot_download(repo_id=str(repo_id), local_dir=repo_local_path)

    # Ensure LFS tracking for mp4
    with Path(repo_local_path / ".gitattributes").open("a+") as f:
        f.seek(0)
        if not any("*.mp4" in line for line in f):
            f.write("*.mp4 filter=lfs diff=lfs merge=lfs -text\n")

    # Step 1: Save the model
    print("Saving model to:", repo_local_path / str(model_name))
    model.save(repo_local_path / str(model_name))

    maybe_vec_normalize = unwrap_vec_normalize(eval_env)
    if maybe_vec_normalize is not None:
        maybe_vec_normalize.save(str(repo_local_path / "vec_normalize.pkl"))
        maybe_vec_normalize.training = False
        maybe_vec_normalize.norm_reward = False

    # Unzip the model for browsing on the Hub
    with zipfile.ZipFile(repo_local_path / model_name.filename, "r") as zip_ref:
        zip_ref.extractall(repo_local_path / str(model_name))

    # Step 2: Copy config files
    args_path = log_path / str(env_name) / "args.yml"
    config_path = log_path / str(env_name) / "config.yml"
    shutil.copy(args_path, repo_local_path / "args.yml")
    shutil.copy(config_path, repo_local_path / "config.yml")
    with Path(repo_local_path / "env_kwargs.yml").open("w") as outfile:
        yaml.dump(env_kwargs, outfile)

    # Bundle train/eval metrics into a zip
    with zipfile.ZipFile(repo_local_path / "train_eval_metrics.zip", "w") as archive:
        evaluations_npz = log_path / "evaluations.npz"
        if evaluations_npz.is_file():
            archive.write(evaluations_npz, arcname="evaluations.npz")
        for monitor_file in (str(p) for p in Path(log_path).glob("*.csv")):
            archive.write(monitor_file, arcname=Path(monitor_file).name)

    # Step 3: Evaluate
    mean_reward, std_reward = _evaluate_agent(model, eval_env, n_eval_episodes, is_deterministic, repo_local_path)

    # Step 4: Video
    if generate_video:
        _generate_replay(model, eval_env, video_length, is_deterministic, repo_local_path)
        video_path = Path("test.mp4")
        if video_path.is_file():
            video_path.unlink()
        json_paths = list(Path().glob("*.meta.json"))
        for jp in json_paths:
            jp.unlink()

    # Step 5: Model card
    generated_model_card, metadata = generate_model_card(
        algo_name,
        algo_class_name,
        _organization,
        env_name.gym_id,
        mean_reward,
        std_reward,
        hyperparams,
        env_kwargs,
    )
    save_model_card(repo_local_path, generated_model_card, metadata)

    console.print(f"Pushing repo {repo_name} to the HuggingFace Hub", style="bold blue")
    api.upload_folder(repo_id=str(repo_id), folder_path=repo_local_path, commit_message=commit_message)

    console.print(f"Your model is pushed to the hub. You can view your model here: {repo_url}", style="bold blue")
    return str(repo_url)


def push_to_hub(
    repo_id: str,
    filename: str,
    commit_message: str | None = None,
    token: str | None = None,
) -> str:
    """Upload a single file to the HuggingFace Hub.

    Adapted from
    https://github.com/huggingface/huggingface_sb3/blob/main/huggingface_sb3/push_to_hub.py

    Args:
        repo_id: Repository id (``"{org}/{name}"``)).
        filename: Local path to the file to upload.
        commit_message: Commit message.
        token: HuggingFace authentication token.

    Returns:
        Repository URL.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    if not _HF_AVAILABLE:
        raise ImportError("huggingface_hub is required to push to the Hub. Install with: pip install huggingface_hub")

    api = HfApi(token=token)
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    console.print(f"Pushing '{filename}' to '{repo_id}' repo on the Hugging Face Hub", style="bold blue")
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=Path(filename).name,
        repo_id=repo_url.repo_id,
        commit_message=commit_message,
    )

    console.print(f"Your model has been uploaded to the Hub, you can find it here: {repo_url}", style="bold green")
    return str(repo_url)
