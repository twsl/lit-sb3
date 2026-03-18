from collections import deque
from collections.abc import Iterable
import glob
from itertools import zip_longest
import os
from pathlib import Path
import platform
import random
import re
from typing import Optional, Union
import warnings

import cloudpickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

import lit_sb3

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from lit_sb3.core.types import GymEnv, Schedule, TensorDict, TrainFreq, TrainFrequencyUnit
from lit_sb3.utils.logger import Logger, configure


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """Seed the different random generators.

    Args:
        seed: Random seed for reproducibility.
        using_cuda: Whether CUDA is being used.
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes fraction of variance that ypred explains about y.

    Returns 1 - Var[y-ypred] / Var[y].

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    Args:
        y_pred: the prediction
        y_true: the expected value

    Returns:
        explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Update the learning rate for a given optimizer.

    Useful when doing linear schedule.

    Args:
        optimizer: Pytorch optimizer
        learning_rate: New learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


class FloatSchedule:
    """Wrapper that ensures the output of a Schedule is cast to float.

    Can wrap either a constant value or an existing callable Schedule.

    Args:
        value_schedule: Constant value or callable schedule
            (e.g. LinearSchedule, ConstantSchedule)
    """

    def __init__(self, value_schedule: Schedule | float) -> None:
        if isinstance(value_schedule, FloatSchedule):
            self.value_schedule: Schedule = value_schedule.value_schedule
        elif isinstance(value_schedule, (float, int)):
            self.value_schedule = ConstantSchedule(float(value_schedule))
        else:
            assert callable(value_schedule), (
                f"The learning rate schedule must be a float or a callable, not {value_schedule}"
            )
            self.value_schedule = value_schedule

    def __call__(self, progress_remaining: float) -> float:
        # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
        # Some types are have odd behaviors when part of a Schedule, like numpy floats
        return float(self.value_schedule(progress_remaining))

    def __repr__(self) -> str:
        return f"FloatSchedule({self.value_schedule})"


class LinearSchedule:
    """LinearSchedule interpolates linearly between start and end.

    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    Args:
        start: value to start with if ``progress_remaining`` = 1
        end: value to end with if ``progress_remaining`` = 0
        end_fraction: fraction of ``progress_remaining``  where end is reached e.g 0.1
            then end is reached after 10% of the complete training process.
    """

    def __init__(self, start: float, end: float, end_fraction: float) -> None:
        self.start = start
        self.end = end
        self.end_fraction = end_fraction

    def __call__(self, progress_remaining: float) -> float:
        if (1 - progress_remaining) > self.end_fraction:
            return self.end
        else:
            return self.start + (1 - progress_remaining) * (self.end - self.start) / self.end_fraction

    def __repr__(self) -> str:
        return f"LinearSchedule(start={self.start}, end={self.end}, end_fraction={self.end_fraction})"


class ConstantSchedule:
    """Constant schedule that always returns the same value.

    Useful for fixed learning rates or clip ranges.

    Args:
        val: constant value
    """

    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, _: float) -> float:
        return self.val

    def __repr__(self) -> str:
        return f"ConstantSchedule(val={self.val})"


# ===== Deprecated schedule functions ====
# only kept for backward compatibility when unpickling old models, use FloatSchedule
# and other classes like `LinearSchedule() instead


def get_schedule_fn(value_schedule: Schedule | float) -> Schedule:
    """Transform (if needed) learning rate and clip range (for PPO).

    to callable.

    Args:
        value_schedule: Constant value of schedule function

    Returns:
        Schedule function (can return constant value)
    """
    warnings.warn("get_schedule_fn() is deprecated, please use FloatSchedule() instead", stacklevel=2)
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
    # Some types are have odd behaviors when part of a Schedule, like numpy floats
    return lambda progress_remaining: float(value_schedule(progress_remaining))


def get_linear_fn(start: float, end: float, end_fraction: float) -> Schedule:
    """Create a function that interpolates linearly between start and end.

    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.

    Returns:
        Linear schedule function.
    """
    warnings.warn("get_linear_fn() is deprecated, please use LinearSchedule() instead", stacklevel=2)

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_fn(val: float) -> Schedule:
    """Create a function that returns a constant.

    It is useful for learning rate schedule (to avoid code duplication).

    Args:
        val: constant value

    Returns:
        Constant schedule function.
    """
    warnings.warn("constant_fn() is deprecated, please use ConstantSchedule() instead", stacklevel=2)

    def func(_: float) -> float:
        return val

    return func


# ==== End of deprecated schedule functions ====


def get_device(device: torch.device | str = "auto") -> torch.device:
    """Retrieve PyTorch device.

    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    Args:
        device: One for 'auto', 'cuda', 'cpu'

    Returns:
        Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def get_latest_run_id(log_path: str | Path = "", log_name: str = "") -> int:
    """Returns the latest run number for the given log name and log path,.

    by finding the greatest number in the directories.

    Args:
        log_path: Path to the log folder containing several runs.
        log_name: Name of the experiment. Each run is stored
            in a folder named ``log_name_1``, ``log_name_2``, ...

    Returns:
        latest run number
    """
    max_run_id = 0
    for path in Path(log_path).glob(f"{glob.escape(log_name)}_[0-9]*"):
        file_name = Path(path).name
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def configure_logger(
    verbose: int = 0,
    tensorboard_log: str | Path | None = None,
    tb_log_name: str = "",
    reset_num_timesteps: bool = True,
) -> Logger:
    """Configure the logger's outputs.

    Args:
        verbose: Verbosity level: 0 for no output, 1 for the standard output to be part of the logger outputs
        tensorboard_log: the log location for tensorboard (if None, no logging)
        tb_log_name: tensorboard log
        reset_num_timesteps: Whether the ``num_timesteps`` attribute is reset or not.
            It allows to continue a previous learning curve (``reset_num_timesteps=False``)
            or start from t=0 (``reset_num_timesteps=True``, the default).

    Returns:
        The logger object
    """
    save_path, format_strings = None, ["stdout"]

    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = str(Path(tensorboard_log) / f"{tb_log_name}_{latest_run_id + 1}")
        format_strings = ["stdout", "tensorboard"] if verbose >= 1 else ["tensorboard"]
    elif verbose == 0:
        format_strings = [""]
    return configure(save_path, format_strings=format_strings)


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
    # We first assume that all observations are not vectorized
    all_non_vectorized = True
    for key, subspace in observation_space.spaces.items():
        # This fails when the observation is not vectorized
        # or when it has the wrong shape
        if observation[key].shape != subspace.shape:
            all_non_vectorized = False
            break

    if all_non_vectorized:
        return False

    all_vectorized = True
    # Now we check that all observation are vectorized and have the correct shape
    for key, subspace in observation_space.spaces.items():
        if observation[key].shape[1:] != subspace.shape:
            all_vectorized = False
            break

    if all_vectorized:
        return True
    else:
        # Retrieve error message
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
        # for-else happens if no break is called
        raise ValueError(
            f"Error: Cannot determine if the observation is vectorized with the space type {observation_space}."
        )


def safe_mean(arr: np.ndarray | list | deque) -> float:
    """Compute the mean of an array if there is at least one element.

    For empty array, return NaN. It is used for logging only.

    Args:
        arr: Numpy array or list of values
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))


def get_parameters_by_name(model: torch.nn.Module, included_names: Iterable[str]) -> list[torch.Tensor]:
    """Extract parameters from the state dict of ``model``.

    if the name contains one of the strings in ``included_names``.

    Args:
        model: the model where the parameters come from.
        included_names: substrings of names to include.

    Returns:
        List of parameters values (Pytorch tensors)
        that matches the queried names.
    """
    return [param for name, param in model.state_dict().items() if any(key in name for key in included_names)]


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""``zip()`` function but enforces that iterables are of equal length.

    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    params: Iterable[torch.Tensor],
    target_params: Iterable[torch.Tensor],
    tau: float,
) -> None:
    """Perform a Polyak average update on ``target_params`` using ``params``.

    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93.

    Args:
        params: parameters to use to update the target params
        target_params: parameters to update
        tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def obs_as_tensor(obs: np.ndarray | dict[str, np.ndarray], device: torch.device) -> torch.Tensor | TensorDict:
    """Moves the observation to the given device.

    Args:
        obs: The input observation array(s).
        device: PyTorch device

    Returns:
        PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: torch.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise TypeError(f"Unrecognized type of observation {type(obs)}")


def should_collect_more_steps(
    train_freq: TrainFreq,
    num_collected_steps: int,
    num_collected_episodes: int,
) -> bool:
    """Helper used in ``collect_rollouts()`` of off-policy algorithms.

    to determine the termination condition.

    Args:
        train_freq: How much experience should be collected before updating the policy.
        num_collected_steps: The number of already collected steps.
        num_collected_episodes: The number of already collected episodes.

    Returns:
        Whether to continue or not collecting experience
        by doing rollouts of the current policy.
    """
    if train_freq.unit == TrainFrequencyUnit.STEP:
        return num_collected_steps < train_freq.frequency

    elif train_freq.unit == TrainFrequencyUnit.EPISODE:
        return num_collected_episodes < train_freq.frequency

    else:
        raise ValueError(
            "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
            f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
        )


def get_system_info(print_info: bool = True) -> tuple[dict[str, str], str]:
    """Retrieve system and python env info for the current system.

    Args:
        print_info: Whether to print or not those infos

    Returns:
        Dictionary summing up the version for each relevant package
        and a formatted string.
    """
    env_info = {
        # In OS, a regex is used to add a space between a "#" and a number to avoid
        # wrongly linking to another issue on GitHub. Example: turn "#42" to "# 42".
        "OS": re.sub(r"#(\d)", r"# \1", f"{platform.platform()} {platform.version()}"),
        "Python": platform.python_version(),
        "lit-sb3": lit_sb3.__version__,
        "PyTorch": torch.__version__,
        "GPU Enabled": str(torch.cuda.is_available()),
        "Numpy": np.__version__,
        "Cloudpickle": cloudpickle.__version__,
        "Gymnasium": gym.__version__,
    }
    try:
        import gym as openai_gym  # type: ignore[import-untyped]

        env_info.update({"OpenAI Gym": openai_gym.__version__})
    except ImportError:
        pass

    env_info_str = ""
    for key, value in env_info.items():
        env_info_str += f"- {key}: {value}\n"
    if print_info:
        print(env_info_str)
    return env_info, env_info_str


# --- Merged from contrib/common/utils.py ---
from collections.abc import Callable, Sequence

from torch import nn


def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    cum_prob: torch.Tensor | None = None,
    sum_over_quantiles: bool = True,
) -> torch.Tensor:
    """The quantile-regression loss, as described in the QR-DQN and TQC papers.

    Partially taken from https://github.com/bayesgroup/tqc_pytorch.

    Args:
        current_quantiles: current estimate of quantiles, must be either
            (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
        target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
            (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
        cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
            must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics,
                n_quantiles).
            (if None, calculating unit quantiles)
        sum_over_quantiles: if summing over the quantile dimension or not

    Returns:
        the loss
    """
    if current_quantiles.ndim != target_quantiles.ndim:
        raise ValueError(
            f"Error: The dimension of curremt_quantile ({current_quantiles.ndim}) needs to match "
            f"the dimension of target_quantiles ({target_quantiles.ndim})."
        )
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError(
            f"Error: The batch size of curremt_quantile ({current_quantiles.shape[0]}) needs to match "
            f"the batch size of target_quantiles ({target_quantiles.shape[0]})."
        )
    if current_quantiles.ndim not in (2, 3):
        raise ValueError(
            f"Error: The dimension of current_quantiles ({current_quantiles.ndim}) needs to be either 2 or 3."
        )

    if cum_prob is None:
        n_quantiles = current_quantiles.shape[-1]
        # Cumulative probabilities to calculate quantiles.
        cum_prob = (torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
        if current_quantiles.ndim == 2:
            # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, -1, 1)
        elif current_quantiles.ndim == 3:
            # For TQC, current_quantiles have a shape (batch_size, n_critics, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_critics, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, 1, -1, 1)

    # QR-DQN
    # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)
    # TQC
    # target_quantiles: (batch_size, 1, n_target_quantiles) -> (batch_size, 1, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_critics, n_quantiles) -> (batch_size, n_critics, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_critics, n_quantiles, n_target_quantiles)
    # Note: in both cases, the loss has the same shape as pairwise_delta
    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    loss = loss.sum(dim=-2).mean() if sum_over_quantiles else loss.mean()
    return loss


def conjugate_gradient_solver(
    matrix_vector_dot_fn: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    max_iter: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Finds an approximate solution to a set of linear equations Ax = b.

    Sources:
     - https://github.com/ajlangley/trpo-pytorch/blob/master/conjugate_gradient.py
     - https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py#L122

    Reference:
     - https://epubs.siam.org/doi/abs/10.1137/1.9781611971446.ch6

    Args:
        matrix_vector_dot_fn: a function that right multiplies a matrix A by a vector v
        b: the right hand term in the set of linear equations Ax = b
        max_iter: the maximum number of iterations (default is 10)
        residual_tol: residual tolerance for early stopping of the solving (default is 1e-10)
            :return x:
            the approximate solution to the system of equations defined by `matrix_vector_dot_fn`
            and b
    """
    # The vector is not initialized at 0 because of the instability issues when the gradient becomes small.
    # A small random gaussian noise is used for the initialization.
    x = 1e-4 * torch.randn_like(b)
    residual = b - matrix_vector_dot_fn(x)
    # Equivalent to torch.linalg.norm(residual) ** 2 (L2 norm squared)
    residual_squared_norm = torch.matmul(residual, residual)

    if residual_squared_norm < residual_tol:
        # If the gradient becomes extremely small
        # The denominator in alpha will become zero
        # Leading to a division by zero
        return x

    p = residual.clone()

    for i in range(max_iter):
        # A @ p (matrix vector multiplication)
        a_dot_p = matrix_vector_dot_fn(p)

        alpha = residual_squared_norm / p.dot(a_dot_p)
        x += alpha * p

        if i == max_iter - 1:
            return x

        residual -= alpha * a_dot_p
        new_residual_squared_norm = torch.matmul(residual, residual)

        if new_residual_squared_norm < residual_tol:
            return x

        beta = new_residual_squared_norm / residual_squared_norm
        residual_squared_norm = new_residual_squared_norm
        p = residual + beta * p
    # Note: this return statement is only used when max_iter=0
    return x


def flat_grad(
    output: torch.Tensor,
    parameters: Sequence[nn.parameter.Parameter],
    create_graph: bool = False,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Returns the gradients of the passed sequence of parameters into a flat gradient.

    Order of parameters is preserved.

    Args:
        output: functional output to compute the gradient for
        parameters: sequence of ``Parameter``
        retain_graph: If ``False``, the graph used to compute the grad will be freed.
            Defaults to the value of ``create_graph``.
        create_graph: If ``True``, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: ``False``.

    Returns:
        Tensor containing the flattened gradients
    """
    grads = torch.autograd.grad(
        output,
        parameters,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat([torch.ravel(grad) for grad in grads if grad is not None])
