import numpy as np

from lit_sb3.core.types import GymEnv
from lit_sb3.vec_env import VecEnv

EXPECTED_METHOD_NAME = "action_masks"


def get_action_masks(env: GymEnv) -> np.ndarray:
    """Checks whether gym env exposes a method returning invalid action masks.

    Args:
        env: the Gym environment to get masks from

    Returns:
        A numpy array of the masks
    """
    if isinstance(env, VecEnv):
        return np.stack(env.env_method(EXPECTED_METHOD_NAME))
    else:
        return env.get_wrapper_attr(EXPECTED_METHOD_NAME)()


def is_masking_supported(env: GymEnv) -> bool:
    """Checks whether gym env exposes a method returning invalid action masks.

    Args:
        env: the Gym environment to check

    Returns:
        True if the method is found, False otherwise
    """
    if isinstance(env, VecEnv):
        return env.has_attr(EXPECTED_METHOD_NAME)
    else:
        try:
            env.get_wrapper_attr(EXPECTED_METHOD_NAME)
        except AttributeError:
            return False
        else:
            return True
