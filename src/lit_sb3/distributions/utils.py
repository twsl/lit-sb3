"""Utility functions for probability distributions."""

from typing import Any

from gymnasium import spaces
import numpy as np
import torch

from lit_sb3.core.preprocessing import get_action_dim
from lit_sb3.distributions.base import Distribution
from lit_sb3.distributions.bernoulli import BernoulliDistribution
from lit_sb3.distributions.categorical import CategoricalDistribution
from lit_sb3.distributions.gaussian import DiagGaussianDistribution
from lit_sb3.distributions.multi_categorical import MultiCategoricalDistribution
from lit_sb3.distributions.sde import StateDependentNoiseDistribution
from lit_sb3.distributions.squashed_gaussian import SquashedDiagGaussianDistribution


def make_proba_distribution(
    action_space: spaces.Space, use_sde: bool = False, dist_kwargs: dict[str, Any] | None = None
) -> Distribution:
    """Return an instance of Distribution for the correct type of action space.

    Args:
        action_space: the input action space
        use_sde: Force the use of StateDependentNoiseDistribution
            instead of DiagGaussianDistribution
        dist_kwargs: Keyword arguments to pass to the probability distribution

    Returns:
        the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(int(action_space.n), **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(list(action_space.nvec), **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        )
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


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
