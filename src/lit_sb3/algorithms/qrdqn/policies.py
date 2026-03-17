from typing import Any, Optional

from gymnasium import spaces
import torch
from torch import nn

from lit_sb3.common.types import PyTorchObs, Schedule
from lit_sb3.modules import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from lit_sb3.policies import BasePolicy


class QuantileNetwork(BasePolicy):
    """Quantile network for QR-DQN.

    Args:
        observation_space: Observation space
        action_space: Action space
        n_quantiles: Number of quantiles
        net_arch: The specification of the network architecture.
        activation_fn: Activation function
        normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        n_quantiles: int = 200,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.n_quantiles = n_quantiles
        action_dim = int(self.action_space.n)  # number of actions
        quantile_net = create_mlp(self.features_dim, action_dim * self.n_quantiles, self.net_arch, self.activation_fn)
        self.quantile_net = nn.Sequential(*quantile_net)

    def forward(self, obs: PyTorchObs) -> torch.Tensor:
        """Predict the quantiles.

        Args:
            obs: Observation

        Returns:
            The estimated quantiles for each action.
        """
        quantiles = self.quantile_net(self.extract_features(obs, self.features_extractor))
        return quantiles.view(-1, self.n_quantiles, int(self.action_space.n))

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> torch.Tensor:
        q_values = self(observation).mean(dim=1)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            {
                "net_arch": self.net_arch,
                "features_dim": self.features_dim,
                "n_quantiles": self.n_quantiles,
                "activation_fn": self.activation_fn,
                "features_extractor": self.features_extractor,
            }
        )
        return data


class QRDQNPolicy(BasePolicy):
    """Policy class with quantile and target networks for QR-DQN.

    Args:
        observation_space: Observation space
        action_space: Action space
        lr_schedule: Learning rate schedule (could be constant)
        n_quantiles: Number of quantiles
        net_arch: The specification of the network architecture.
        activation_fn: Activation function
        features_extractor_class: Features extractor to use.
        features_extractor_kwargs: Keyword arguments
            to pass to the features extractor.
        normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
        optimizer_class: The optimizer to use,
            ``torch.optim.Adam`` by default
        optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
    """

    quantile_net: QuantileNetwork
    quantile_net_target: QuantileNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [] if features_extractor_class == NatureCNN else [64, 64]

        self.n_quantiles = n_quantiles
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_quantiles": self.n_quantiles,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """Create the network and the optimizer.

        Args:
            lr_schedule: Learning rate schedule
                lr_schedule(1) is the initial learning rate
        """
        self.quantile_net = self.make_quantile_net()
        self.quantile_net_target = self.make_quantile_net()
        self.quantile_net_target.load_state_dict(self.quantile_net.state_dict())
        self.quantile_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.quantile_net.parameters(),
            lr=lr_schedule(1),  # ty:ignore[unknown-argument]
            **self.optimizer_kwargs,
        )

    def make_quantile_net(self) -> QuantileNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QuantileNetwork(**net_args).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        return self.quantile_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            {
                "n_quantiles": self.net_args["n_quantiles"],
                "net_arch": self.net_args["net_arch"],
                "activation_fn": self.net_args["activation_fn"],
                "lr_schedule": self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                "optimizer_class": self.optimizer_class,
                "optimizer_kwargs": self.optimizer_kwargs,
                "features_extractor_class": self.features_extractor_class,
                "features_extractor_kwargs": self.features_extractor_kwargs,
            }
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        Args:
            mode: if true, set to training mode, else set to evaluation mode
        """
        self.quantile_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = QRDQNPolicy


class CnnPolicy(QRDQNPolicy):
    """Policy class for QR-DQN when using images as input.

    Args:
        observation_space: Observation space
        action_space: Action space
        lr_schedule: Learning rate schedule (could be constant)
        n_quantiles: Number of quantiles
        net_arch: The specification of the network architecture.
        activation_fn: Activation function
        features_extractor_class: Features extractor to use.
        normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
        optimizer_class: The optimizer to use,
            ``torch.optim.Adam`` by default
        optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_quantiles,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputPolicy(QRDQNPolicy):
    """Policy class for QR-DQN when using dict observations as input.

    Args:
        observation_space: Observation space
        action_space: Action space
        lr_schedule: Learning rate schedule (could be constant)
        n_quantiles: Number of quantiles
        net_arch: The specification of the network architecture.
        activation_fn: Activation function
        features_extractor_class: Features extractor to use.
        normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
        optimizer_class: The optimizer to use,
            ``torch.optim.Adam`` by default
        optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_quantiles,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
