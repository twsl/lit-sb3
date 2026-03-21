from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import ResizeObservation


class YAMLCompatResizeObservation(ResizeObservation):
    """ResizeObservation that accepts a list instead of a tuple for YAML compatibility."""

    def __init__(self, env: gym.Env, shape: list[int]) -> None:
        super().__init__(env, (shape[0], shape[1]))
