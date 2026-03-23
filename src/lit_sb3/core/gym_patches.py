"""Patches for gymnasium so lit-sb3 keeps working as expected.

Notably patches the TimeLimit wrapper to keep the behaviour as before and
provide additional info that the episode reached a timeout, but only when
the episode is over because of that.

Adapted from
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/gym_patches.py
"""

from typing import Any

import numpy as np

# Deprecation warning with gym 0.26 and numpy 1.24
np.bool8 = np.bool_  # type: ignore[attr-defined]

import gymnasium  # noqa: E402


class PatchedTimeLimit(gymnasium.wrappers.TimeLimit):
    """Patched TimeLimit that sets ``TimeLimit.truncated`` correctly.

    See https://github.com/openai/gym/issues/3102 and
    https://github.com/Farama-Foundation/Gymnasium/pull/101:
    keep the behaviour as before and provide additional info
    that the episode reached a timeout, but only when the episode
    is over because of that.
    """

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1  # ty:ignore[unsupported-operator]

        if self._elapsed_steps >= self._max_episode_steps:
            done = truncated or terminated
            # TimeLimit.truncated key may have been already set by the environment;
            # do not overwrite it, only set it when the episode is not over for
            # other reasons.
            episode_truncated = not done or info.get("TimeLimit.truncated", False)
            info["TimeLimit.truncated"] = episode_truncated
            # truncated may have been set by the env too
            truncated = truncated or episode_truncated

        return observation, reward, terminated, truncated, info


# Patch Gymnasium TimeLimit
gymnasium.wrappers.TimeLimit = PatchedTimeLimit  # type: ignore[misc]
gymnasium.wrappers.common.TimeLimit = PatchedTimeLimit  # type: ignore[misc]
