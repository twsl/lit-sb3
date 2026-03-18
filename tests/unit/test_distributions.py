"""Unit tests for probability distributions.

Adapted from stable-baselines3 tests/test_distributions.py.
"""

from copy import deepcopy

import pytest
import torch as th

from lit_sb3.core.utils import set_random_seed
from lit_sb3.distributions.bernoulli import BernoulliDistribution
from lit_sb3.distributions.categorical import CategoricalDistribution
from lit_sb3.distributions.gaussian import DiagGaussianDistribution
from lit_sb3.distributions.multi_categorical import MultiCategoricalDistribution
from lit_sb3.distributions.sde import StateDependentNoiseDistribution
from lit_sb3.distributions.squashed_gaussian import SquashedDiagGaussianDistribution
from lit_sb3.distributions.tanh_bijector import TanhBijector
from lit_sb3.distributions.utils import kl_divergence

N_ACTIONS = 2
N_FEATURES = 3
N_SAMPLES = int(5e4)  # reduced from 5e6 for test speed


def test_bijector():
    """Test TanhBijector."""
    actions = th.ones(5) * 2.0
    bijector = TanhBijector()

    squashed_actions = bijector.forward(actions)
    assert th.max(th.abs(squashed_actions)) <= 1.0
    assert th.isclose(TanhBijector.inverse(squashed_actions), actions).all()


def test_squashed_gaussian_bounds():
    """Test that SquashedDiagGaussianDistribution keeps actions within [-1, 1]."""
    set_random_seed(1)
    gaussian_mean = th.rand(N_SAMPLES, N_ACTIONS)
    dist = SquashedDiagGaussianDistribution(N_ACTIONS)
    _, log_std = dist.proba_distribution_net(N_FEATURES)
    dist = dist.proba_distribution(gaussian_mean, log_std)
    actions = dist.get_actions()
    assert th.max(th.abs(actions)) <= 1.0


@pytest.mark.parametrize(
    "dist",
    [
        DiagGaussianDistribution(N_ACTIONS),
        StateDependentNoiseDistribution(N_ACTIONS, squash_output=False),
    ],
)
def test_entropy(dist):
    """Differential entropy approximated by mean negative log likelihood."""
    set_random_seed(1)
    deterministic_actions = th.rand(1, N_ACTIONS).repeat(N_SAMPLES, 1)
    _, log_std = dist.proba_distribution_net(N_FEATURES, log_std_init=th.log(th.tensor(0.2)))

    if isinstance(dist, DiagGaussianDistribution):
        dist = dist.proba_distribution(deterministic_actions, log_std)
    else:
        state = th.rand(1, N_FEATURES).repeat(N_SAMPLES, 1)
        dist.sample_weights(log_std, batch_size=N_SAMPLES)
        dist = dist.proba_distribution(deterministic_actions, log_std, state)

    actions = dist.get_actions()
    entropy = dist.entropy()
    log_prob = dist.log_prob(actions)
    assert th.allclose(entropy.mean(), -log_prob.mean(), rtol=5e-2)


@pytest.mark.parametrize(
    "dist, cat_actions",
    [
        (CategoricalDistribution(N_ACTIONS), N_ACTIONS),
        (MultiCategoricalDistribution([2, 3]), sum([2, 3])),
        (BernoulliDistribution(N_ACTIONS), N_ACTIONS),
    ],
)
def test_categorical(dist, cat_actions):
    """Entropy approximated by mean negative log likelihood for categorical types."""
    set_random_seed(1)
    action_logits = th.rand(N_SAMPLES, cat_actions)
    dist = dist.proba_distribution(action_logits)
    actions = dist.get_actions()
    entropy = dist.entropy()
    log_prob = dist.log_prob(actions)
    assert th.allclose(entropy.mean(), -log_prob.mean(), rtol=5e-2)


@pytest.mark.parametrize(
    "dist_type",
    [
        BernoulliDistribution(N_ACTIONS).proba_distribution(th.rand(N_ACTIONS)),
        CategoricalDistribution(N_ACTIONS).proba_distribution(th.rand(N_ACTIONS)),
        DiagGaussianDistribution(N_ACTIONS).proba_distribution(th.rand(N_ACTIONS), th.rand(N_ACTIONS)),
        MultiCategoricalDistribution([N_ACTIONS, N_ACTIONS]).proba_distribution(
            th.rand(1, sum([N_ACTIONS, N_ACTIONS]))
        ),
        SquashedDiagGaussianDistribution(N_ACTIONS).proba_distribution(th.rand(N_ACTIONS), th.rand(N_ACTIONS)),
        StateDependentNoiseDistribution(N_ACTIONS).proba_distribution(
            th.rand(N_ACTIONS), th.rand([N_ACTIONS, N_ACTIONS]), th.rand([N_ACTIONS, N_ACTIONS])
        ),
    ],
)
def test_kl_divergence(dist_type):
    set_random_seed(8)
    # Test 1: same distribution should have KL Div = 0
    dist1 = dist_type
    dist2 = dist_type
    assert th.allclose(kl_divergence(dist1, dist2).sum(), th.tensor(0.0))

    # Test 2: KL Div = E(Unbiased approx KL Div)
    if isinstance(dist_type, CategoricalDistribution):
        dist1 = dist_type.proba_distribution(th.rand(N_ACTIONS).repeat(N_SAMPLES, 1))
        dist2 = deepcopy(dist_type).proba_distribution(th.rand(N_ACTIONS).repeat(N_SAMPLES, 1))
    elif isinstance(dist_type, (DiagGaussianDistribution, SquashedDiagGaussianDistribution)):
        mean_actions1 = th.rand(1).repeat(N_SAMPLES, 1)
        log_std1 = th.rand(1).repeat(N_SAMPLES, 1)
        mean_actions2 = th.rand(1).repeat(N_SAMPLES, 1)
        log_std2 = th.rand(1).repeat(N_SAMPLES, 1)
        dist1 = dist_type.proba_distribution(mean_actions1, log_std1)
        dist2 = deepcopy(dist_type).proba_distribution(mean_actions2, log_std2)
    elif isinstance(dist_type, BernoulliDistribution):
        dist1 = dist_type.proba_distribution(th.rand(1).repeat(N_SAMPLES, 1))
        dist2 = deepcopy(dist_type).proba_distribution(th.rand(1).repeat(N_SAMPLES, 1))
    elif isinstance(dist_type, MultiCategoricalDistribution):
        dist1 = dist_type.proba_distribution(th.rand(1, sum([N_ACTIONS, N_ACTIONS])).repeat(N_SAMPLES, 1))
        dist2 = deepcopy(dist_type).proba_distribution(th.rand(1, sum([N_ACTIONS, N_ACTIONS])).repeat(N_SAMPLES, 1))
    elif isinstance(dist_type, StateDependentNoiseDistribution):
        dist1 = StateDependentNoiseDistribution(1)
        dist2 = deepcopy(dist1)
        state = th.rand(1, N_FEATURES).repeat(N_SAMPLES, 1)
        mean_actions1 = th.rand(1).repeat(N_SAMPLES, 1)
        mean_actions2 = th.rand(1).repeat(N_SAMPLES, 1)
        _, log_std = dist1.proba_distribution_net(N_FEATURES, log_std_init=th.log(th.tensor(0.2)))  # ty:ignore[invalid-argument-type]
        dist1.sample_weights(log_std, batch_size=N_SAMPLES)
        dist2.sample_weights(log_std, batch_size=N_SAMPLES)
        dist1 = dist1.proba_distribution(mean_actions1, log_std, state)
        dist2 = dist2.proba_distribution(mean_actions2, log_std, state)

    full_kl_div = kl_divergence(dist1, dist2).mean(dim=0)
    actions = dist1.get_actions()
    approx_kl_div = (dist1.log_prob(actions) - dist2.log_prob(actions)).mean(dim=0)
    # Use a generous rtol to accommodate MC approximation variance at N_SAMPLES=5e4
    assert th.allclose(full_kl_div, approx_kl_div, rtol=0.15)

    # Test 3: Sanity test with Bernoulli distribution
    if isinstance(dist_type, BernoulliDistribution):
        dist1 = BernoulliDistribution(1).proba_distribution(th.tensor([0.3]))
        dist2 = BernoulliDistribution(1).proba_distribution(th.tensor([0.65]))
        full_kl_div = kl_divergence(dist1, dist2)
        actions = th.tensor([0.0, 1.0])
        ad_hoc_kl = th.sum(
            th.exp(dist1.distribution.log_prob(actions))  # ty:ignore[unresolved-attribute]
            * (dist1.distribution.log_prob(actions) - dist2.distribution.log_prob(actions))  # ty:ignore[unresolved-attribute]
        )
        assert th.allclose(full_kl_div, ad_hoc_kl)


def test_sde_distribution():
    n_actions = 1
    deterministic_actions = th.ones(N_SAMPLES, n_actions) * 0.1
    state = th.ones(N_SAMPLES, N_FEATURES) * 0.3
    dist = StateDependentNoiseDistribution(n_actions, full_std=True, squash_output=False)

    set_random_seed(1)
    _, log_std = dist.proba_distribution_net(N_FEATURES)
    dist.sample_weights(log_std, batch_size=N_SAMPLES)
    dist = dist.proba_distribution(deterministic_actions, log_std, state)
    actions = dist.get_actions()

    assert th.allclose(actions.mean(), dist.distribution.mean.mean(), rtol=2e-2)  # ty:ignore[unresolved-attribute]
    assert th.allclose(actions.std(), dist.distribution.scale.mean(), rtol=2e-2)  # ty:ignore[unresolved-attribute]
