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

from lit_sb3.zoo.utils import (
    ALGOS,
    EnvironmentName,
    ModelName,
    SimpleLinearSchedule,
    create_test_env,
    get_callback_list,
    get_class_by_name,
    get_hf_trained_models,
    get_latest_run_id,
    get_model_path,
    get_saved_hyperparams,
    get_trained_models,
    get_wrapper_class,
    linear_schedule,
)

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
