"""Argument parser and logic for the ``record-video`` sub-command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

from lit_sb3.core.utils import set_random_seed
from lit_sb3.vec_env import VecVideoRecorder
from lit_sb3.zoo.exp_manager import ExperimentManager
from lit_sb3.zoo.utils import ALGOS, EnvironmentName, StoreDict, create_test_env, get_model_path, get_saved_hyperparams


def _add_record_video_args(parser: argparse.ArgumentParser) -> None:
    """Add record-video arguments to *parser*."""
    parser.add_argument("--env", help="Environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str)
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ppo",
        type=str,
        required=False,
        choices=list(ALGOS.keys()),
    )
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps", default=1000, type=int)
    parser.add_argument("--n-envs", help="Number of environments", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--no-render",
        action="store_true",
        default=False,
        help="Do not render the environment (useful for tests)",
    )
    parser.add_argument(
        "--exp-id",
        help="Experiment ID (default: 0: latest, -1: no exp folder)",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--load-best",
        action="store_true",
        default=False,
        help="Load best model instead of last model if available",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor",
    )
    parser.add_argument(
        "--custom-objects",
        action="store_true",
        default=False,
        help="Use custom objects to solve loading issues",
    )


def run_record_video(args: argparse.Namespace) -> None:  # noqa: C901
    """Execute record-video from a parsed :class:`argparse.Namespace`."""
    import lit_sb3.zoo.import_envs  # noqa: F401

    env_name: EnvironmentName = EnvironmentName(args.env)
    algo = args.algo
    folder = args.folder
    video_folder = args.output_folder
    seed = args.seed
    video_length = args.n_timesteps
    n_envs = args.n_envs

    name_prefix, model_path, log_path = get_model_path(
        args.exp_id,
        folder,
        algo,
        env_name,
        args.load_best,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )

    print(f"Loading {model_path}")
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc", "crossq"]

    set_random_seed(args.seed)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = Path(log_path) / str(env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path)

    env_kwargs: dict[str, Any] = {}
    args_path = Path(log_path) / str(env_name) / "args.yml"
    if Path(args_path).is_file():
        with Path(args_path).open() as f:
            loaded_args = yaml.safe_load(f)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    env_kwargs.update(render_mode="rgb_array")

    env = create_test_env(
        env_name.gym_id,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs: dict[str, Any] = {"seed": args.seed}
    if algo in off_policy_algos:
        kwargs["buffer_size"] = 1
        if "optimize_memory_usage" in hyperparams:
            kwargs["optimize_memory_usage"] = False

    newer_python_version = sys.version_info >= (3, 8)

    custom_objects: dict[str, Any] = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    stochastic = args.stochastic or ((is_atari or is_minigrid) and not args.deterministic)
    deterministic = not stochastic

    if video_folder is None:
        video_folder = Path(log_path) / "videos"

    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=name_prefix,
    )

    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    try:
        for _ in range(video_length):
            action, lstm_states = model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            if not args.no_render:
                env.render()
            obs, _, dones, _ = env.step(action)
            episode_starts = dones
    except KeyboardInterrupt:
        pass

    env.close()
