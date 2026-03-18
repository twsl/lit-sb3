"""Main CLI dispatcher for lit-sb3.

Dispatches to ``train``, ``enjoy``, ``load_from_hub``, ``push_to_hub``,
``plot_train``, ``plot_from_file`` and ``all_plots`` sub-commands.

Usage::

    python -m lit_sb3 train --algo ppo --env CartPole-v1
    python -m lit_sb3 enjoy --algo ppo --env CartPole-v1
    python -m lit_sb3 load_from_hub --algo ppo --env CartPole-v1 -f logs/ -orga sb3
    python -m lit_sb3 push_to_hub --algo ppo --env CartPole-v1 -f logs/ -orga my-org
    python -m lit_sb3 plot_train -a ppo -e CartPole-v1 -f logs/
"""

from __future__ import annotations

import argparse
import sys

from lit_sb3.core.cli import (
    _add_all_plots_args,
    _add_enjoy_args,
    _add_load_from_hub_args,
    _add_plot_from_file_args,
    _add_plot_train_args,
    _add_push_to_hub_args,
    _add_train_args,
    run_all_plots,
    run_enjoy,
    run_load_from_hub,
    run_plot_from_file,
    run_plot_train,
    run_push_to_hub,
    run_train,
)


def main() -> None:
    """Dispatch to the requested sub-command."""
    if len(sys.argv) < 2:
        print("Usage: lit-sb3 <command> [args...]")
        print("Available commands: train, enjoy, load_from_hub, push_to_hub, plot_train, plot_from_file, all_plots")
        sys.exit(1)

    script_name = sys.argv[1]
    # Remove script name from argv so the subcommand parser sees clean args
    del sys.argv[1]

    _COMMANDS: dict[str, tuple] = {
        "train": (
            argparse.ArgumentParser(description="Train an RL agent."),
            _add_train_args,
            run_train,
        ),
        "enjoy": (
            argparse.ArgumentParser(description="Run a trained RL agent."),
            _add_enjoy_args,
            run_enjoy,
        ),
        "load_from_hub": (
            argparse.ArgumentParser(description="Download a pre-trained agent from the HuggingFace Hub."),
            _add_load_from_hub_args,
            run_load_from_hub,
        ),
        "push_to_hub": (
            argparse.ArgumentParser(description="Push a trained model to the HuggingFace Hub."),
            _add_push_to_hub_args,
            run_push_to_hub,
        ),
        "plot_train": (
            argparse.ArgumentParser(description="Plot training reward and success rate."),
            _add_plot_train_args,
            run_plot_train,
        ),
        "plot_from_file": (
            argparse.ArgumentParser(description="Plot results from a saved pickle file."),
            _add_plot_from_file_args,
            run_plot_from_file,
        ),
        "all_plots": (
            argparse.ArgumentParser(description="Aggregate and plot results from multiple experiments."),
            _add_all_plots_args,
            run_all_plots,
        ),
    }

    if script_name not in _COMMANDS:
        raise ValueError(f"The command {script_name!r} is unknown, please use one of {list(_COMMANDS.keys())}")

    parser, add_args, run = _COMMANDS[script_name]
    add_args(parser)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
