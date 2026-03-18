"""Argument parser and logic for the ``push-to-hub`` sub-command."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch as th
import yaml

from lit_sb3.core.utils import set_random_seed
from lit_sb3.zoo.utils import ALGOS, StoreDict, create_test_env, get_model_path, get_saved_hyperparams


def _add_push_to_hub_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", help="Environment ID", type=str, required=True)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, required=True)
    parser.add_argument("--algo", help="RL Algorithm", type=str, required=True, choices=list(ALGOS.keys()))
    parser.add_argument(
        "-n", "--n-timesteps", help="Number of timesteps for the video recording", default=1000, type=int
    )
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="Number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render",
        action="store_true",
        default=False,
        help="Do not render the environment (useful for tests)",
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best",
        action="store_true",
        default=False,
        help="Load best model instead of last model if available",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help=(
            "Load checkpoint instead of last model if available, "
            "you must pass the number of timesteps corresponding to it"
        ),
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor",
    )
    parser.add_argument("-orga", "--organization", help="HuggingFace hub organization", type=str, required=True)
    parser.add_argument(
        "-name",
        "--repo-name",
        help="HuggingFace hub repository name (default: '{algo}-{env}')",
        type=str,
    )
    parser.add_argument("-m", "--commit-message", help="Commit message", default="Initial commit", type=str)


def run_push_to_hub(args: argparse.Namespace) -> None:
    """Execute the push-to-hub command with pre-parsed args."""
    from lit_sb3.huggingface.naming_schemes import EnvironmentName, ModelName, ModelRepoId
    from lit_sb3.huggingface.push_to_hub import package_to_hub
    from lit_sb3.zoo.exp_manager import ExperimentManager
    import lit_sb3.zoo.import_envs  # noqa: F401

    env_name: EnvironmentName = EnvironmentName(args.env)
    algo = args.algo

    _, model_path, log_path = get_model_path(
        args.exp_id,
        args.folder,
        algo,
        env_name,
        args.load_best,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )

    print(f"Loading {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc", "crossq"]
    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = Path(log_path) / str(env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, test_mode=True)

    env_kwargs: dict[str, Any] = {}
    args_path = Path(log_path) / str(env_name) / "args.yml"
    if Path(args_path).is_file():
        with Path(args_path).open() as f:
            loaded_args = yaml.safe_load(f)
            if loaded_args.get("env_kwargs") is not None:
                env_kwargs = loaded_args["env_kwargs"]

    should_render = not args.no_render
    if should_render:
        env_kwargs.update(render_mode="rgb_array")

    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    eval_env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=None,
        should_render=should_render,
        hyperparams=deepcopy(hyperparams),
        env_kwargs=env_kwargs,
    )

    kwargs: dict[str, Any] = {"seed": args.seed}
    if algo in off_policy_algos:
        kwargs.update({"buffer_size": 1})
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    custom_objects: dict[str, Any] = {}
    model = ALGOS[algo].load(model_path, env=eval_env, custom_objects=custom_objects, device=args.device, **kwargs)

    stochastic = args.stochastic or ((is_atari or is_minigrid) and not args.deterministic)
    deterministic = not stochastic

    model_name = ModelName(algo, env_name)
    if args.repo_name is None:
        args.repo_name = str(model_name)

    repo_id = ModelRepoId(args.organization, args.repo_name)
    print(f"Uploading to {args.organization}/{args.repo_name}, make sure to have the rights")

    package_to_hub(
        model,
        model_name,
        algo,
        ALGOS[algo].__name__,
        Path(log_path),
        hyperparams,
        env_kwargs,
        env_name,
        eval_env,
        repo_id=repo_id,
        commit_message=args.commit_message,
        is_deterministic=deterministic,
        n_eval_episodes=10,
        token=None,
        local_repo_path="hub",
        video_length=args.n_timesteps,
        generate_video=not args.no_render,
    )
