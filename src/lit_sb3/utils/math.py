"""Mathematical utilities: statistics, losses, optimisation helpers."""

from collections import deque
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from lit_sb3.distributions.base import Distribution
from lit_sb3.distributions.multi_categorical import MultiCategoricalDistribution


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


def safe_mean(arr: np.ndarray | list | deque) -> float:
    """Compute the mean of an array if there is at least one element.

    For empty array, return NaN. It is used for logging only.

    Args:
        arr: Numpy array or list of values
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))


def kl_divergence(dist_true: Distribution, dist_pred: Distribution) -> torch.Tensor:
    """Wrapper for the PyTorch implementation of the full form KL Divergence.

    Args:
        dist_true: the p distribution
        dist_pred: the q distribution

    Returns:
        KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert dist_true.__class__ == dist_pred.__class__, "Error: input distributions should be the same type"

    # MultiCategoricalDistribution is not a PyTorch Distribution subclass
    # so we need to implement it ourselves!
    if isinstance(dist_pred, MultiCategoricalDistribution):
        assert isinstance(dist_true, MultiCategoricalDistribution)  # already checked above, for mypy
        assert np.allclose(dist_pred.action_dims, dist_true.action_dims), (
            f"Error: distributions must have the same input space: {dist_pred.action_dims} != {dist_true.action_dims}"
        )
        return torch.stack(
            [
                torch.distributions.kl_divergence(p, q)
                for p, q in zip(dist_true.distribution, dist_pred.distribution, strict=False)
            ],
            dim=1,
        ).sum(dim=1)

    # Use the PyTorch kl_divergence implementation
    else:
        return torch.distributions.kl_divergence(dist_true.distribution, dist_pred.distribution)  # ty:ignore[invalid-argument-type]


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
        cum_prob = (torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
        if current_quantiles.ndim == 2:
            cum_prob = cum_prob.view(1, -1, 1)
        elif current_quantiles.ndim == 3:
            cum_prob = cum_prob.view(1, 1, -1, 1)

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
    x = 1e-4 * torch.randn_like(b)
    residual = b - matrix_vector_dot_fn(x)
    residual_squared_norm = torch.matmul(residual, residual)

    if residual_squared_norm < residual_tol:
        return x

    p = residual.clone()

    for i in range(max_iter):
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
    return x
