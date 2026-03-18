"""Zoo plots — re-export plot functions from core/cli.

Adapted from
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/plots/__init__.py
"""

from lit_sb3.core.cli.all_plots import all_plots
from lit_sb3.core.cli.plot_from_file import plot_from_file, restyle_boxplot
from lit_sb3.core.cli.plot_train import plot_train

__all__ = [
    "all_plots",
    "plot_from_file",
    "plot_train",
    "restyle_boxplot",
]
