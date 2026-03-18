"""Argument-parsing helpers and logic runners for each lit-sb3 CLI sub-command."""

from lit_sb3.core.cli.all_plots import _add_all_plots_args, run_all_plots
from lit_sb3.core.cli.enjoy import _add_enjoy_args, run_enjoy
from lit_sb3.core.cli.load_from_hub import _add_load_from_hub_args, run_load_from_hub
from lit_sb3.core.cli.plot_from_file import _add_plot_from_file_args, run_plot_from_file
from lit_sb3.core.cli.plot_train import _add_plot_train_args, run_plot_train
from lit_sb3.core.cli.push_to_hub import _add_push_to_hub_args, run_push_to_hub
from lit_sb3.core.cli.record_training import _add_record_training_args, run_record_training
from lit_sb3.core.cli.record_video import _add_record_video_args, run_record_video
from lit_sb3.core.cli.train import _add_train_args, run_train

__all__ = [
    "_add_all_plots_args",
    "_add_enjoy_args",
    "_add_load_from_hub_args",
    "_add_plot_from_file_args",
    "_add_plot_train_args",
    "_add_push_to_hub_args",
    "_add_record_training_args",
    "_add_record_video_args",
    "_add_train_args",
    "run_all_plots",
    "run_enjoy",
    "run_load_from_hub",
    "run_plot_from_file",
    "run_plot_train",
    "run_push_to_hub",
    "run_record_training",
    "run_record_video",
    "run_train",
]
