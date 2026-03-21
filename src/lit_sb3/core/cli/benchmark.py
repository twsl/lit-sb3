"""Argument parser and logic for the ``benchmark`` sub-command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

import lit_sb3.zoo.import_envs  # noqa: F401
from lit_sb3.zoo.utils import get_latest_run_id, get_saved_hyperparams, get_trained_models

try:
    import pandas as pd
    import pytablewriter  # ty:ignore[unresolved-import]

    _TABLE_DEPS_AVAILABLE = True
except ImportError:
    _TABLE_DEPS_AVAILABLE = False

try:
    from lit_sb3.core.results_plotter import load_results, ts2xy

    _RESULTS_PLOTTER_AVAILABLE = True
except ImportError:
    _RESULTS_PLOTTER_AVAILABLE = False

try:
    from lit_sb3.huggingface.load_from_hub import download_from_hub

    _HUB_DOWNLOAD_AVAILABLE = True
except ImportError:
    _HUB_DOWNLOAD_AVAILABLE = False


def _add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add benchmark arguments to *parser*."""
    parser.add_argument("--log-dir", help="Root log folder", default="rl-trained-agents/", type=str)
    parser.add_argument("--benchmark-dir", help="Benchmark log folder", default="logs/benchmark/", type=str)
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=150000, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Do only one experiment (useful for testing)",
    )
    parser.add_argument("--with-mujoco", action="store_true", default=False, help="Run also MuJoCo envs")
    parser.add_argument("--no-hub", action="store_true", default=False, help="Do not download models from hub")
    parser.add_argument("--num-threads", help="Number of threads for PyTorch", default=2, type=int)


def run_benchmark(args: argparse.Namespace) -> None:  # noqa: C901
    """Run benchmark evaluation over all trained models and generate a report."""
    trained_models = get_trained_models(args.log_dir)

    if not args.no_hub:
        try:
            from lit_sb3.zoo.utils import get_hf_trained_models

            trained_models.update(get_hf_trained_models())
        except (ImportError, Exception) as e:
            print(f"Warning: Could not fetch HuggingFace models: {e}")

    n_experiments = len(trained_models)
    results: dict[str, list] = {
        "algo": [],
        "env_id": [],
        "mean_reward": [],
        "std_reward": [],
        "n_timesteps": [],
        "eval_timesteps": [],
        "eval_episodes": [],
    }

    for idx, trained_model in enumerate(trained_models.keys()):
        algo, env_id = trained_models[trained_model]

        n_envs = args.n_envs
        n_timesteps = args.n_timesteps

        # HER is now a replay buffer class
        if algo == "her":
            continue

        if algo in ["dqn", "qrdqn", "ddpg", "sac", "td3", "tqc", "crossq"]:
            n_envs = 1
            n_timesteps *= args.n_envs

        reward_log = Path(args.benchmark_dir) / trained_model
        arguments = [
            "-n",
            str(n_timesteps),
            "--n-envs",
            str(n_envs),
            "-f",
            args.log_dir,
            "--algo",
            algo,
            "--env",
            env_id,
            "--no-render",
            "--num-threads",
            str(args.num_threads),
            "--seed",
            str(args.seed),
            "--verbose",
            "0",
            "--reward-log",
            str(reward_log),
        ]
        if args.verbose >= 1:
            print(f"{idx + 1}/{n_experiments}")
            print(f"Evaluating {algo} on {env_id}...")

        skip_eval = False
        if Path(reward_log).is_dir() and _RESULTS_PLOTTER_AVAILABLE:
            try:
                x, y = ts2xy(load_results(reward_log), "timesteps")
                skip_eval = len(x) > 0
            except (json.JSONDecodeError, Exception):  # noqa: S110
                pass

        # Comment out to benchmark HER robotics env — requires a mujoco licence
        if "Fetch" in env_id and not args.with_mujoco:
            print(f"Skipping mujoco env: {env_id}")
            if not skip_eval:
                continue

        if skip_eval:
            print("Skipping eval...")
        else:
            return_code = subprocess.call(["python", "-m", "lit_sb3", "enjoy", *arguments])  # noqa: S603, S607
            if return_code != 0:
                print("Error during evaluation, skipping...")
                continue

            if not _RESULTS_PLOTTER_AVAILABLE:
                print("Warning: lit_sb3.core.results_plotter not available, skipping result loading")
                continue

            x, y = ts2xy(load_results(reward_log), "timesteps")

        if not _RESULTS_PLOTTER_AVAILABLE:
            continue

        if len(x) > 0:
            # Retrieve training timesteps from config
            exp_id = get_latest_run_id(Path(args.log_dir) / algo, env_id)
            log_path = Path(args.log_dir) / algo / f"{env_id}_{exp_id}" / env_id
            hyperparams, _ = get_saved_hyperparams(log_path)
            # Model not downloaded
            if len(hyperparams) == 0 and _HUB_DOWNLOAD_AVAILABLE:
                print(
                    "Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3"
                )
                exp_id = 1
                log_path = Path(args.log_dir) / algo / f"{env_id}_{exp_id}" / env_id
                # Auto-download
                try:
                    download_from_hub(
                        algo=algo,
                        env_name=env_id,  # type: ignore[arg-type]
                        exp_id=exp_id,
                        folder=args.log_dir,
                        organization="sb3",
                        repo_name=None,
                        force=False,
                    )
                    # Try again
                    hyperparams, _ = get_saved_hyperparams(log_path)
                except Exception as e:
                    print(f"Failed to download: {e}")

            # Hack to format it properly
            if hyperparams.get("n_timesteps", 0) < 1e6:
                n_training_timesteps = f"{int(hyperparams.get('n_timesteps', 0) / 1e3)}k"
            else:
                n_training_timesteps = f"{int(hyperparams.get('n_timesteps', 0) / 1e6)}M"

            mean_reward = np.mean(y)
            std_reward = np.std(y)
            results["algo"].append(algo)
            results["env_id"].append(env_id)
            results["mean_reward"].append(mean_reward)
            results["std_reward"].append(std_reward)
            results["n_timesteps"].append(n_training_timesteps)
            results["eval_timesteps"].append(x[-1])
            results["eval_episodes"].append(len(y))
            if args.verbose >= 1:
                print(x[-1], "timesteps")
                print(len(y), "Episodes")
                print(f"Mean reward: {mean_reward:.2f} +- {std_reward:.2f}")
                print()
        else:
            print("Not enough timesteps")

        if args.test_mode:
            break

    if not _TABLE_DEPS_AVAILABLE:
        print("Warning: pandas and pytablewriter are required to generate the benchmark table.")
        print("Install them with: pip install pandas pytablewriter")
        return

    # Create DataFrame
    results_df = pd.DataFrame(results)
    # Sort results
    results_df = results_df.sort_values(by=["algo", "env_id"])

    header = """
## Performance of trained agents

Final performance of the trained agents can be found in the table below.
This was computed by running `python -m lit_sb3 benchmark`:
it runs the trained agent (trained on `n_timesteps`)
for `eval_timesteps` and then reports the mean episode reward
during this evaluation.

It uses the deterministic policy except for Atari games.

*NOTE: this is not a quantitative benchmark as it corresponds to only one run
(cf [issue #38](https://github.com/araffin/rl-baselines-zoo/issues/38)).
This benchmark is meant to check algorithm (maximal) performance, find potential bugs
and also allow users to have access to pretrained agents.*

"M" stands for Million (1e6)

"""

    writer = pytablewriter.MarkdownTableWriter(max_precision=3)
    writer.from_dataframe(results_df)

    Path(args.benchmark_dir).mkdir(parents=True, exist_ok=True)
    tmp_path = Path(args.benchmark_dir) / "benchmark.md"
    with Path(tmp_path).open("w") as f:
        f.write(header)
        writer.stream = f
        writer.write_table()
    print(f"Results written to: {tmp_path}")

    # Update root benchmark file
    if not args.test_mode:
        shutil.copy(tmp_path, "benchmark.md")
        print("Results copied to: benchmark.md")
