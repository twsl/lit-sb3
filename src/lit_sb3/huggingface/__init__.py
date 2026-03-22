"""HuggingFace Hub integration for lit-sb3.

Integrates code from:

- https://github.com/huggingface/huggingface_sb3/tree/main/huggingface_sb3
- https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/

All imports adapted to use ``lit_sb3`` instead of ``stable_baselines3``.
"""

from __future__ import annotations

from lit_sb3.huggingface.naming_schemes import EnvironmentName, ModelName, ModelRepoId

__all__ = [
    "EnvironmentName",
    "ModelName",
    "ModelRepoId",
]
