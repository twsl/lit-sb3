"""Utility function for creating maskable probability distributions."""

from gymnasium import spaces

from lit_sb3.distributions.maskable.base import MaskableDistribution
from lit_sb3.distributions.maskable.bernoulli import MaskableBernoulliDistribution
from lit_sb3.distributions.maskable.categorical import MaskableCategoricalDistribution
from lit_sb3.distributions.maskable.multi_categorical import MaskableMultiCategoricalDistribution


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
