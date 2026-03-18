"""Maskable Bernoulli distribution for multi-binary action spaces."""

from lit_sb3.distributions.maskable.multi_categorical import MaskableMultiCategoricalDistribution


class MaskableBernoulliDistribution(MaskableMultiCategoricalDistribution):
    """Bernoulli distribution for multibinary actions. Supports invalid action masking.

    Args:
        action_dim: Number of binary actions
    """

    def __init__(self, action_dim: int) -> None:
        # Two states per binary action
        action_dims = [2] * action_dim
        super().__init__(action_dims)
