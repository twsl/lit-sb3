"""lit_sb3.zoo — training/evaluation utilities adapted from RL Baselines3 Zoo.

Provides an :class:`ExperimentManager` that reads YAML hyperparameter files,
creates environments and callbacks, and drives ``model.learn()`` (which
delegates to :class:`~lit_sb3.trainer.RLTrainer` internally).

Quickstart::

    from lit_sb3.zoo import ExperimentManager
    from lit_sb3.zoo.utils import ALGOS

CLI entry points:
    lit-sb3  →  lit_sb3.cli:main  (e.g. ``lit-sb3 train``, ``lit-sb3 enjoy``)
"""

# isort: off
import lit_sb3.zoo.gym_patches  # noqa: F401
# isort: on

from lit_sb3.huggingface.naming_schemes import EnvironmentName, ModelName
from lit_sb3.utils.env import create_test_env
from lit_sb3.utils.model import get_hf_trained_models, get_model_path, get_saved_hyperparams, get_trained_models
from lit_sb3.utils.reflection import get_callback_list, get_class_by_name, get_wrapper_class
from lit_sb3.utils.registry import ALGOS
from lit_sb3.utils.schedule import SimpleLinearSchedule, linear_schedule
from lit_sb3.utils.system import get_latest_run_id

__all__ = [
    "ALGOS",
    "EnvironmentName",
    "ModelName",
    "SimpleLinearSchedule",
    "create_test_env",
    "get_callback_list",
    "get_class_by_name",
    "get_hf_trained_models",
    "get_latest_run_id",
    "get_model_path",
    "get_saved_hyperparams",
    "get_trained_models",
    "get_wrapper_class",
    "linear_schedule",
]
