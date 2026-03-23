"""Score normalization utilities for Pybullet environment benchmarks.

Adapted from
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/plots/score_normalization.py
"""

from __future__ import annotations

import numpy as np

# Minimum and maximum scores (from random and expert agents) for Pybullet envs
# These are used to normalize scores to [0, 1] range.
MIN_SCORE: dict[str, float] = {
    "HalfCheetahBulletEnv-v0": -2000.0,
    "AntBulletEnv-v0": -300.0,
    "HopperBulletEnv-v0": 16.0,
    "Walker2DBulletEnv-v0": 1.3,
    "LunarLanderContinuous-v3": -200.0,
    "BipedalWalker-v3": -100.0,
}

MAX_SCORE: dict[str, float] = {
    "HalfCheetahBulletEnv-v0": 3000.0,
    "AntBulletEnv-v0": 3500.0,
    "HopperBulletEnv-v0": 2500.0,
    "Walker2DBulletEnv-v0": 2500.0,
    "LunarLanderContinuous-v3": 200.0,
    "BipedalWalker-v3": 300.0,
}


def normalize_score(scores: np.ndarray, env_id: str) -> np.ndarray:
    """Normalize scores to [0, 1] range using min/max for the environment.

    :param scores: Array of scores.
    :param env_id: Environment ID.
    :return: Normalised scores.
    :raises KeyError: If ``env_id`` is not in the normalisation tables.
    """
    min_score = MIN_SCORE[env_id]
    max_score = MAX_SCORE[env_id]
    return (scores - min_score) / (max_score - min_score)
