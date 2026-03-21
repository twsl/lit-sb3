"""Policies: abstract base class and concrete implementations."""

from abc import ABC, abstractmethod
import collections
import copy
from functools import partial
from typing import Any, Optional, TypeVar, Union
import warnings

from gymnasium import spaces
import numpy as np
import torch
from torch import nn

from lit_sb3.core.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from lit_sb3.core.types import PyTorchObs, Schedule
from lit_sb3.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from lit_sb3.modules import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from lit_sb3.utils.device import get_device
from lit_sb3.utils.distribution import make_proba_distribution
from lit_sb3.utils.spaces import is_vectorized_observation
from lit_sb3.utils.tensors import obs_as_tensor

SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")


class BaseModel(nn.Module):
    """The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    Args:
        observation_space: The observation space of the environment
        action_space: The action space of the environment
        features_extractor_class: Features extractor to use.
        features_extractor_kwargs: Keyword arguments
            to pass to the features extractor.
        features_extractor: Network to extract features
            (a CNN when using images, a nn.Flatten() layer otherwise)
        normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
        optimizer_class: The optimizer to use,
            ``torch.optim.Adam`` by default
        optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
    """

    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        features_extractor: BaseFeaturesExtractor | None = None,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        # Automatically deactivate dtype and bounds checks
        if not normalize_images and issubclass(features_extractor_class, (NatureCNN, CombinedExtractor)):
            self.features_extractor_kwargs.update({"normalized_image": True})

    def _update_features_extractor(
        self,
        net_kwargs: dict[str, Any],
        features_extractor: BaseFeaturesExtractor | None = None,
    ) -> dict[str, Any]:
        """Update the network keyword arguments and create a new features extractor object if needed.

        If a ``features_extractor`` object is passed, then it will be shared.

        Args:
            net_kwargs: the base network keyword arguments, without the ones
                related to features extractor
            features_extractor: a features extractor object.
                If None, a new object will be created.

        Returns:
            The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor()
        net_kwargs.update({"features_extractor": features_extractor, "features_dim": features_extractor.features_dim})
        return net_kwargs

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def extract_features(self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor) -> torch.Tensor:
        """Preprocess the observation if needed and extract features.

        Args:
            obs: Observation
            features_extractor: The features extractor to use.

        Returns:
            The extracted features
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Get data that need to be saved in order to re-create the model when loading it from disk.

        Returns:
            The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            "normalize_images": self.normalize_images,
        }

    @property
    def device(self) -> torch.device:
        """Infer which device this policy lives on by inspecting its parameters.

        If it has no parameters, the 'cpu' device is used as a fallback.
        """
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def save(self, path: str) -> None:
        """Save model to a given location.

        Args:
            path:
            path: Path to save the policy to.
        """
        torch.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls: type[SelfBaseModel], path: str, device: torch.device | str = "auto") -> SelfBaseModel:
        """Load model from path.

        Args:
            path:
            device: Device on which the policy should be loaded.
            path: Path to save the policy to.
        """
        device = get_device(device)
        # Note(antonin): we cannot use `weights_only=True` here because we need to allow
        # gymnasium imports for the policy to be loaded successfully
        saved_variables = torch.load(path, map_location=device, weights_only=False)

        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """Load parameters from a 1D vector.

        Args:
            vector:
            vector: The vector parameter.
        """
        torch.nn.utils.vector_to_parameters(
            torch.as_tensor(vector, dtype=torch.float, device=self.device), self.parameters()
        )

    def parameters_to_vector(self) -> np.ndarray:
        """Convert the parameters to a 1D vector."""
        return torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def set_training_mode(self, mode: bool) -> None:
        """Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        Args:
            mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def is_vectorized_observation(self, observation: np.ndarray | dict[str, np.ndarray]) -> bool:
        """Check whether or not the observation is vectorized,.

        apply transposition to image (so that they are channel-first) if needed.
        This is used in DQN when sampling random action (epsilon-greedy policy).

        Args:
            observation: the input observation to check

        Returns:
            whether the given observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict), (
                f"The observation provided is a dict but the obs space is {self.observation_space}"
            )
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]  # ty:ignore[invalid-argument-type]
                vectorized_env = vectorized_env or is_vectorized_observation(maybe_transpose(obs, obs_space), obs_space)  # ty:ignore[invalid-argument-type]
        else:
            vectorized_env = is_vectorized_observation(
                maybe_transpose(observation, self.observation_space), self.observation_space
            )
        return vectorized_env

    def obs_to_tensor(self, observation: np.ndarray | dict[str, np.ndarray]) -> tuple[PyTorchObs, bool]:
        """Convert an input observation to a PyTorch tensor that can be fed to a model.

        Includes sugar-coating to handle different observations (e.g. normalizing images).

        Args:
            observation: the input observation

        Returns:
            The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict), (
                f"The observation provided is a dict but the obs space is {self.observation_space}"
            )
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]  # ty:ignore[invalid-argument-type]
                obs_ = maybe_transpose(obs, obs_space) if is_image_space(obs_space) else np.array(obs)  # ty:ignore[invalid-argument-type]
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))  # type: ignore[misc]

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, vectorized_env


class BasePolicy(BaseModel, ABC):
    """The base policy object.

    Parameters are mostly the same as `BaseModel`; additions are documented below.

    Args:
        args: positional arguments passed through to `BaseModel`.
        kwargs: keyword arguments passed through to `BaseModel`.
        squash_output: For continuous actions, whether the output is squashed
            or not using a ``tanh()`` function.
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(self, *args, squash_output: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """Orthogonal initialization (used in PPO and A2C)."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        """Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        Args:
            observation:
            deterministic: Whether to use stochastic or deterministic actions

            observation: The current observation.

        Returns:
            Taken action according to the policy
        """

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Get the policy action from an observation (and optional hidden state).

        Includes sugar-coating to handle different observations (e.g. normalizing images).

        Args:
            observation: the input observation
            state: The last hidden states (can be None, used in recurrent policies)
            episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            deterministic: Whether or not to return deterministic actions.

        Returns:
            the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with torch.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale the action from [low, high] to [-1, 1].

        (no need for symmetric action space).

        Args:
            action: Action to scale

        Returns:
            Scaled action
        """
        assert isinstance(self.action_space, spaces.Box), (
            f"Trying to scale an action using an action space that is not a Box(): {self.action_space}"
        )
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """Rescale the action from [-1, 1] to [low, high].

        (no need for symmetric action space).

        Args:
            scaled_action: Action to un-scale
        """
        assert isinstance(self.action_space, spaces.Box), (
            f"Trying to unscale an action using an action space that is not a Box(): {self.action_space}"
        )
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
