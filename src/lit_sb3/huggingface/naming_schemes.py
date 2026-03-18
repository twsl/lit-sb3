"""Naming conventions for HuggingFace Hub model repositories.

Adapted from
https://github.com/huggingface/huggingface_sb3/blob/main/huggingface_sb3/naming_schemes.py

This module provides string subclasses that enforce a uniform naming scheme for
repository ids, making it safe to build file paths and URLs from environment /
model names.
"""

from __future__ import annotations

# Note: best practice for immutable str subclasses is to use __new__.
# https://stackoverflow.com/a/2673863


class EnvironmentName(str):
    """A normalized environment name safe for use in file paths and URLs.

    Slashes are replaced by dashes, and ``namespace:EnvName`` style ids are
    stripped of their namespace prefix so that:

    .. code-block:: python

        EnvironmentName("ALE/Pong-v5")  # → "ALE-Pong-v5"
        EnvironmentName("ns:CartPole-v1")  # → "CartPole-v1"

    The original gym id (to pass to ``gym.make``) is available via the
    :attr:`gym_id` property.
    """

    _gym_id: str

    def __new__(cls, gym_id: str) -> EnvironmentName:
        normalized_str = gym_id.replace("/", "-")
        if ":" in normalized_str:
            split_by_colon = normalized_str.split(":")
            if len(split_by_colon) == 2:
                # strip namespace prefix before the colon
                normalized_str = split_by_colon[1]
            else:
                raise ValueError(f"Environment name {gym_id!r} contains more than one colon!")
        instance = super().__new__(cls, normalized_str)
        instance._gym_id = gym_id
        return instance

    @property
    def gym_id(self) -> str:
        """The original gym id to pass to ``gym.make``."""
        return self._gym_id


class ModelName(str):
    """A model name derived from algorithm and environment.

    Safe to use in file paths and URLs because a normalized
    :class:`EnvironmentName` is used.

    .. code-block:: python

        ModelName("ppo", EnvironmentName("CartPole-v1"))  # → "ppo-CartPole-v1"
    """

    def __new__(cls, algo_name: str, environment_name: EnvironmentName) -> ModelName:
        return super().__new__(cls, f"{algo_name}-{environment_name}")

    @property
    def filename(self) -> str:
        """The filename under which the model is stored when calling ``model.save``."""
        return f"{self}.zip"


class ModelRepoId(str):
    """The name of a HuggingFace repository.

    Derived from the associated organization and the model name.

    .. code-block:: python

        ModelRepoId("sb3", ModelName("ppo", EnvironmentName("CartPole-v1")))
        # → "sb3/ppo-CartPole-v1"
    """

    def __new__(cls, org: str, model_name: str) -> ModelRepoId:
        return super().__new__(cls, f"{org}/{model_name}")
