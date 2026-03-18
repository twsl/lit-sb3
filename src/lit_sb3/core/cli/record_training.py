"""Argument parser and logic for the ``record-training`` sub-command."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import subprocess

from lit_sb3.zoo.utils import ALGOS, EnvironmentName, get_latest_run_id


def _add_record_training_args(parser: argparse.ArgumentParser) -> None:
    """Add record-training arguments to *parser*."""
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
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
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("-g", "--gif", action="store_true", default=False, help="Convert final video to gif")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--exp-id",
        help="Experiment ID (default: 0: latest, -1: no exp folder)",
        default=0,
        type=int,
    )


def run_record_training(args: argparse.Namespace) -> None:  # noqa: C901
    """Record training videos from checkpoints and stitch them together."""
    env_name: EnvironmentName = EnvironmentName(args.env)
    algo = args.algo
    folder = args.folder
    video_folder = args.output_folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(Path(folder) / algo, env_name)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    log_path = Path(folder) / algo / f"{env_name}_{args.exp_id}" if args.exp_id > 0 else Path(folder) / algo

    assert Path(log_path).is_dir(), f"The {log_path} folder was not found"

    if video_folder is None:
        video_folder = Path().resolve()
