"""Maskable probability distributions for reinforcement learning with invalid action masking."""

from lit_sb3.distributions.maskable.base import MaskableDistribution, MaybeMasks
from lit_sb3.distributions.maskable.bernoulli import MaskableBernoulliDistribution
from lit_sb3.distributions.maskable.categorical import (
    MaskableCategoricalDistribution,
    SelfMaskableCategoricalDistribution,
)
from lit_sb3.distributions.maskable.maskable_categorical import MaskableCategorical
from lit_sb3.distributions.maskable.multi_categorical import (
    MaskableMultiCategoricalDistribution,
    SelfMaskableMultiCategoricalDistribution,
)
from lit_sb3.distributions.maskable.utils import make_masked_proba_distribution

__all__ = [
    "MaybeMasks",
    "MaskableDistribution",
    "MaskableCategorical",
    "MaskableCategoricalDistribution",
    "SelfMaskableCategoricalDistribution",
    "MaskableMultiCategoricalDistribution",
    "SelfMaskableMultiCategoricalDistribution",
    "MaskableBernoulliDistribution",
    "make_masked_proba_distribution",
]
