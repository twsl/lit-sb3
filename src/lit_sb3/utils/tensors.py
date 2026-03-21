"""PyTorch tensor and parameter utilities."""

from collections.abc import Iterable, Sequence
from itertools import zip_longest

import numpy as np
import torch
from torch import nn

from lit_sb3.core.types import TensorDict


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
