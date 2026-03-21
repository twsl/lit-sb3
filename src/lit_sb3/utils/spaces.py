"""Gymnasium observation-space shape validation helpers."""

from gymnasium import spaces
import numpy as np

from lit_sb3.core.types import GymEnv


def check_for_correct_spaces(env: GymEnv, observation_space: spaces.Space, action_space: spaces.Space) -> None:
    """Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if.

    spaces match after loading the model with given env.
    Checked parameters:
    - observation_space
    - action_space.

    Args:
        env: Environment to check for valid spaces
        observation_space: Observation space to check against
        action_space: Action space to check against
    """
    if observation_space != env.observation_space:
        raise ValueError(f"Observation spaces do not match: {observation_space} != {env.observation_space}")
    if action_space != env.action_space:
        raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")


def check_shape_equal(space1: spaces.Space, space2: spaces.Space) -> None:
    """If the spaces are Box, check that they have the same shape.

    If the spaces are Dict, it recursively checks the subspaces.

    Args:
        space1: Space
        space2: Other space
    """
    if isinstance(space1, spaces.Dict):
        assert isinstance(space2, spaces.Dict), f"spaces must be of the same type: {type(space1)} != {type(space2)}"
        assert space1.spaces.keys() == space2.spaces.keys(), (
            f"spaces must have the same keys: {list(space1.spaces.keys())} != {list(space2.spaces.keys())}"
        )
        for key in space1.spaces:
            check_shape_equal(space1.spaces[key], space2.spaces[key])
    elif isinstance(space1, spaces.Box):
        assert space1.shape == space2.shape, f"spaces must have the same shape: {space1.shape} != {space2.shape}"


def is_vectorized_box_observation(observation: np.ndarray, observation_space: spaces.Box) -> bool:
    """For box observation type, detects and validates the shape,.

    then returns whether or not the observation is vectorized.

    Args:
        observation: the input observation to validate
        observation_space: the observation space

    Returns:
        whether the given observation is vectorized or not
    """
    if observation.shape == observation_space.shape:
        return False
    elif observation.shape[1:] == observation_space.shape:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            f"Box environment, please use {observation_space.shape} "
            + "or (n_env, {}) for the observation shape.".format(", ".join(map(str, observation_space.shape)))
        )


def is_vectorized_discrete_observation(observation: int | np.ndarray, observation_space: spaces.Discrete) -> bool:
    """For discrete observation type, detects and validates the shape,.

    then returns whether or not the observation is vectorized.

    Args:
        observation: the input observation to validate
        observation_space: the observation space

    Returns:
        whether the given observation is vectorized or not
    """
    if isinstance(observation, int) or observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
        return False
    elif len(observation.shape) == 1:
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for "
            "Discrete environment, please use () or (n_env,) for the observation shape."
        )


def is_vectorized_multidiscrete_observation(observation: np.ndarray, observation_space: spaces.MultiDiscrete) -> bool:
    """For multidiscrete observation type, detects and validates the shape,.

    then returns whether or not the observation is vectorized.

    Args:
        observation: the input observation to validate
        observation_space: the observation space

    Returns:
        whether the given observation is vectorized or not
    """
    if observation.shape == (len(observation_space.nvec),):
        return False
    elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
            f"environment, please use ({len(observation_space.nvec)},) or "
            f"(n_env, {len(observation_space.nvec)}) for the observation shape."
        )


def is_vectorized_multibinary_observation(observation: np.ndarray, observation_space: spaces.MultiBinary) -> bool:
    """For multibinary observation type, detects and validates the shape,.

    then returns whether or not the observation is vectorized.

    Args:
        observation: the input observation to validate
        observation_space: the observation space

    Returns:
        whether the given observation is vectorized or not
    """
    if observation.shape == observation_space.shape:
        return False
    elif (
        len(observation.shape) == len(observation_space.shape) + 1 and observation.shape[1:] == observation_space.shape
    ):
        return True
    else:
        raise ValueError(
            f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
            f"environment, please use {observation_space.shape} or "
            f"(n_env, {observation_space.n}) for the observation shape."
        )


def is_vectorized_dict_observation(observation: np.ndarray, observation_space: spaces.Dict) -> bool:
    """For dict observation type, detects and validates the shape,.

    then returns whether or not the observation is vectorized.

    Args:
        observation: the input observation to validate
        observation_space: the observation space

    Returns:
        whether the given observation is vectorized or not
    """
    all_non_vectorized = True
    for key, subspace in observation_space.spaces.items():
        if observation[key].shape != subspace.shape:
            all_non_vectorized = False
            break

    if all_non_vectorized:
        return False

    all_vectorized = True
    for key, subspace in observation_space.spaces.items():
        if observation[key].shape[1:] != subspace.shape:
            all_vectorized = False
            break

    if all_vectorized:
        return True
    else:
        error_msg = ""
        try:
            is_vectorized_observation(observation[key], observation_space.spaces[key])
        except ValueError as e:
            error_msg = f"{e}"
        raise ValueError(
            f"There seems to be a mix of vectorized and non-vectorized observations. "
            f"Unexpected observation shape {observation[key].shape} for key {key} "
            f"of type {observation_space.spaces[key]}. {error_msg}"
        )


def is_vectorized_observation(observation: int | np.ndarray, observation_space: spaces.Space) -> bool:
    """For every observation type, detects and validates the shape,.

    then returns whether or not the observation is vectorized.

    Args:
        observation: the input observation to validate
        observation_space: the observation space

    Returns:
        whether the given observation is vectorized or not
    """
    is_vec_obs_func_dict = {
        spaces.Box: is_vectorized_box_observation,
        spaces.Discrete: is_vectorized_discrete_observation,
        spaces.MultiDiscrete: is_vectorized_multidiscrete_observation,
        spaces.MultiBinary: is_vectorized_multibinary_observation,
        spaces.Dict: is_vectorized_dict_observation,
    }

    for space_type, is_vec_obs_func in is_vec_obs_func_dict.items():
        if isinstance(observation_space, space_type):
            return is_vec_obs_func(observation, observation_space)  # type: ignore[operator]
    else:
        raise ValueError(
            f"Error: Cannot determine if the observation is vectorized with the space type {observation_space}."
        )
