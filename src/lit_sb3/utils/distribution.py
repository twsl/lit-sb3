"""Probability distribution factory functions."""

from typing import Any

from gymnasium import spaces

from lit_sb3.core.preprocessing import get_action_dim
from lit_sb3.distributions.base import Distribution
from lit_sb3.distributions.bernoulli import BernoulliDistribution
from lit_sb3.distributions.categorical import CategoricalDistribution
from lit_sb3.distributions.gaussian import DiagGaussianDistribution
from lit_sb3.distributions.maskable.base import MaskableDistribution
from lit_sb3.distributions.maskable.bernoulli import MaskableBernoulliDistribution
from lit_sb3.distributions.maskable.categorical import MaskableCategoricalDistribution
from lit_sb3.distributions.maskable.multi_categorical import MaskableMultiCategoricalDistribution
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


def make_masked_proba_distribution(action_space: spaces.Space) -> MaskableDistribution:
    """Return an instance of Distribution for the correct type of action space.

    Args:
        action_space: the input action space

    Returns:
        the appropriate Distribution object
    """
    if isinstance(action_space, spaces.Discrete):
        return MaskableCategoricalDistribution(int(action_space.n))
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MaskableMultiCategoricalDistribution(list(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        )
        return MaskableBernoulliDistribution(action_space.n)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Discrete, MultiDiscrete."
        )
