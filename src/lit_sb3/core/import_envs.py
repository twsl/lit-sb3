"""Register optional third-party gym environments and no-velocity variants.

Adapted from
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/import_envs.py

Optional environment packages are imported with graceful fallback so that
missing packages do not break the rest of the package.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from typing import Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from lit_sb3.wrappers import MaskVelocityWrapper

with contextlib.suppress(ImportError):
    import pybullet_envs_gymnasium  # noqa: F401  # ty:ignore[unresolved-import]

try:
    import ale_py

    # no-op: ensure ALE envs are registered
    gym.register_envs(ale_py)
except ImportError:
    pass

try:
    import highway_env  # noqa: F401  # ty:ignore[unresolved-import]
except ImportError:
    pass
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

with contextlib.suppress(ImportError):
    import custom_envs  # noqa: F401  # ty:ignore[unresolved-import]

with contextlib.suppress(ImportError):
    import gym_donkeycar  # noqa: F401  # ty:ignore[unresolved-import]

with contextlib.suppress(ImportError):
    import panda_gym  # noqa: F401  # ty:ignore[unresolved-import]

with contextlib.suppress(ImportError):
    import rocket_lander_gym  # noqa: F401  # ty:ignore[unresolved-import]

with contextlib.suppress(ImportError):
    import minigrid  # noqa: F401  # ty:ignore[unresolved-import]


# ---------------------------------------------------------------------------
# Register "no-velocity" variants of standard environments
# ---------------------------------------------------------------------------


def create_no_vel_env(env_id: str) -> Callable[[str | None], gym.Env]:
    """Return a factory that creates env_id wrapped with :class:`MaskVelocityWrapper`."""

    def make_env(render_mode: str | None = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for _env_id in MaskVelocityWrapper.velocity_indices:
    _name, _version = _env_id.split("-v")
    register(
        id=f"{_name}NoVel-v{_version}",
        entry_point=create_no_vel_env(_env_id),  # type: ignore[arg-type]
    )
