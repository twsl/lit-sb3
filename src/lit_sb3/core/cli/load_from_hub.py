"""Argument parser and logic for the ``load-from-hub`` sub-command."""

from __future__ import annotations

import argparse

from lit_sb3.zoo.utils import ALGOS


def _add_load_from_hub_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", help="Environment ID", type=str, required=True)
    parser.add_argument("-f", "--folder", help="Log folder", type=str, required=True)
    parser.add_argument("-orga", "--organization", help="HuggingFace hub organization", default="sb3", type=str)
    parser.add_argument(
        "-name",
        "--repo-name",
        help="HuggingFace hub repository name (default: '{algo}-{env_id}')",
        type=str,
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        type=str,
        required=True,
        choices=list(ALGOS.keys()),
    )
    parser.add_argument(
        "--exp-id",
        help="Experiment ID (0: auto-increment to latest + 1, -1: no exp folder)",
        default=0,
        type=int,
    )
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Allow overwriting the exp folder if it already exists",
    )


def run_load_from_hub(args: argparse.Namespace) -> None:
    """Execute the load-from-hub command with pre-parsed args."""
    from lit_sb3.huggingface.load_from_hub import download_from_hub
    from lit_sb3.huggingface.naming_schemes import EnvironmentName

    download_from_hub(
        algo=args.algo,
        env_name=EnvironmentName(args.env),
        exp_id=args.exp_id,
        folder=args.folder,
        organization=args.organization,
        repo_name=args.repo_name,
        force=args.force,
    )
