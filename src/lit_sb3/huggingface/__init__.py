"""HuggingFace Hub integration for lit-sb3.

Integrates code from:

- https://github.com/huggingface/huggingface_sb3/tree/main/huggingface_sb3
- https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/

All imports adapted to use ``lit_sb3`` instead of ``stable_baselines3``.
"""

from __future__ import annotations

from lit_sb3.huggingface.load_from_hub import download_from_hub, load_from_hub
from lit_sb3.huggingface.naming_schemes import EnvironmentName, ModelName, ModelRepoId
from lit_sb3.huggingface.push_to_hub import (
    generate_model_card,
    package_to_hub,
    push_to_hub,
    save_model_card,
)

__all__ = [
    "EnvironmentName",
    "ModelName",
    "ModelRepoId",
    "download_from_hub",
    "generate_model_card",
    "load_from_hub",
    "package_to_hub",
    "push_to_hub",
    "save_model_card",
]
