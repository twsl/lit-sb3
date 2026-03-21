"""Probability distributions for reinforcement learning."""

from lit_sb3.distributions.base import Distribution, SelfDistribution, sum_independent_dims
from lit_sb3.distributions.bernoulli import BernoulliDistribution, SelfBernoulliDistribution
from lit_sb3.distributions.categorical import CategoricalDistribution, SelfCategoricalDistribution
from lit_sb3.distributions.gaussian import DiagGaussianDistribution, SelfDiagGaussianDistribution
from lit_sb3.distributions.multi_categorical import MultiCategoricalDistribution, SelfMultiCategoricalDistribution
from lit_sb3.distributions.sde import SelfStateDependentNoiseDistribution, StateDependentNoiseDistribution
from lit_sb3.distributions.squashed_gaussian import (
    SelfSquashedDiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from lit_sb3.distributions.tanh_bijector import TanhBijector

__all__ = [
    "Distribution",
    "SelfDistribution",
    "sum_independent_dims",
    "DiagGaussianDistribution",
    "SelfDiagGaussianDistribution",
    "SquashedDiagGaussianDistribution",
    "SelfSquashedDiagGaussianDistribution",
    "CategoricalDistribution",
    "SelfCategoricalDistribution",
    "MultiCategoricalDistribution",
    "SelfMultiCategoricalDistribution",
    "BernoulliDistribution",
    "SelfBernoulliDistribution",
    "StateDependentNoiseDistribution",
    "SelfStateDependentNoiseDistribution",
    "TanhBijector",
]
