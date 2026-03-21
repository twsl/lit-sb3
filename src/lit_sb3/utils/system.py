"""System information and logging configuration utilities."""

import glob
from pathlib import Path
import platform
import re

import cloudpickle
import gymnasium as gym
import numpy as np
import torch

import lit_sb3

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from lit_sb3.utils.logger import Logger, configure


def get_latest_run_id(log_path: str | Path = "", log_name: str = "") -> int:
    """Returns the latest run number for the given log name and log path,.

    by finding the greatest number in the directories.

    Args:
        log_path: Path to the log folder containing several runs.
        log_name: Name of the experiment. Each run is stored
            in a folder named ``log_name_1``, ``log_name_2``, ...

    Returns:
        latest run number
    """
    max_run_id = 0
    for path in Path(log_path).glob(f"{glob.escape(log_name)}_[0-9]*"):
        file_name = Path(path).name
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def configure_logger(
    verbose: int = 0,
    tensorboard_log: str | Path | None = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
) -> Logger:
    """Configure the logger's outputs.

    Args:
        verbose: Verbosity level: 0 for no output, 1 for the standard output to be part of the logger outputs
        tensorboard_log: the log location for tensorboard (if None, no logging)
        tb_log_name: tensorboard log
        reset_num_timesteps: Whether the ``num_timesteps`` attribute is reset or not.
            It allows to continue a previous learning curve (``reset_num_timesteps=False``)
            or start from t=0 (``reset_num_timesteps=True``, the default).

    Returns:
        The logger object
    """
    save_path, format_strings = None, ["stdout"]

    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = str(Path(tensorboard_log) / f"{tb_log_name}_{latest_run_id + 1}")
        format_strings = ["stdout", "tensorboard"] if verbose >= 1 else ["tensorboard"]
    elif verbose == 0:
        format_strings = [""]
    return configure(save_path, format_strings=format_strings)


def get_system_info(print_info: bool = True) -> tuple[dict[str, str], str]:
    """Retrieve system and python env info for the current system.

    Args:
        print_info: Whether to print or not those infos

    Returns:
        Dictionary summing up the version for each relevant package
        and a formatted string.
    """
    env_info = {
        # In OS, a regex is used to add a space between a "#" and a number to avoid
        # wrongly linking to another issue on GitHub. Example: turn "#42" to "# 42".
        "OS": re.sub(r"#(\d)", r"# \1", f"{platform.platform()} {platform.version()}"),
        "Python": platform.python_version(),
        "lit-sb3": lit_sb3.__version__,
        "PyTorch": torch.__version__,
        "GPU Enabled": str(torch.cuda.is_available()),
        "Numpy": np.__version__,
        "Cloudpickle": cloudpickle.__version__,
        "Gymnasium": gym.__version__,
    }
    try:
        import gym as openai_gym  # type: ignore[import-untyped]

        env_info.update({"OpenAI Gym": openai_gym.__version__})
    except ImportError:
        pass

    env_info_str = ""
    for key, value in env_info.items():
        env_info_str += f"- {key}: {value}\n"
    if print_info:
        print(env_info_str)
    return env_info, env_info_str
