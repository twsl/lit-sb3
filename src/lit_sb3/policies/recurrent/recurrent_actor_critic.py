from typing import Any, Optional, Union

from gymnasium import spaces
import numpy as np
import torch
from torch import nn

from lit_sb3.common.types import RNNStates, Schedule
from lit_sb3.common.utils import zip_strict
from lit_sb3.distributions import Distribution
from lit_sb3.modules import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from lit_sb3.policies.actor_critic import ActorCriticPolicy


class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """Recurrent policy class for actor-critic algorithms (has both policy and value prediction).

    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    Args:
        observation_space: Observation space
        action_space: Action space
        lr_schedule: Learning rate schedule (could be constant)
        net_arch: The specification of the policy and value networks.
        activation_fn: Activation function
        ortho_init: Whether to use or not orthogonal initialization
        use_sde: Whether to use State Dependent Exploration or not
        log_std_init: Initial value for the log standard deviation
        full_std: Whether to use (n_features x n_actions) parameters
            for the std instead of only (n_features,) when using gSDE
        use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
        squash_output: Whether to squash the output using a tanh function,
            this allows to ensure boundaries when using gSDE.
        features_extractor_class: Features extractor to use.
        features_extractor_kwargs: Keyword arguments
            to pass to the features extractor.
        share_features_extractor: If True, the features extractor is shared between the policy and value networks.
        normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
        optimizer_class: The optimizer to use,
            ``torch.optim.Adam`` by default
        optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
        lstm_hidden_size: Number of hidden units for each LSTM layer.
        n_lstm_layers: Number of LSTM layers.
        shared_lstm: Whether the LSTM is shared between the actor and the critic
            (in that case, only the actor gradient is used)
            By default, the actor and the critic have two separate LSTM.
        enable_critic_lstm: Use a seperate LSTM for the critic.
        lstm_kwargs: Additional keyword arguments to pass the the LSTM
            constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: dict[str, Any] | None = None,
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = nn.LSTM(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (self.shared_lstm and self.enable_critic_lstm), (
            "You must choose between shared LSTM, seperate or no LSTM for the critic."
        )

        assert not (self.shared_lstm and not self.share_features_extractor), (
            "If the features extractor is not shared, the LSTM cannot be shared."
        )

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = nn.LSTM(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # ty:ignore[unknown-argument]

    def _build_mlp_extractor(self) -> None:
        """Create the policy and value networks.

        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    @staticmethod
    def _process_sequence(
        features: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        lstm: nn.LSTM,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Do a forward pass in the LSTM network.

        Args:
            features: Input tensor
            lstm_states: previous hidden and cell states of the LSTM, respectively
            episode_starts: Indicates when a new episode starts,
                in that case, we need to reset LSTM states.
            lstm: LSTM object.

        Returns:
            LSTM output and updated LSTM states.
        """
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if torch.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = torch.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = torch.flatten(torch.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states  # ty:ignore[invalid-return-type]

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:  # ty:ignore[invalid-method-override]
        """Forward pass in all the networks (actor and critic).

        Args:
            obs: Observation. Observation
            lstm_states: The last hidden and memory states for the LSTM.
            episode_starts: Whether the observations correspond to new episodes
                or not (we reset the lstm states in that case).
            deterministic: Whether to sample or use deterministic actions

        Returns:
            action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)  # ty:ignore[invalid-argument-type]
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features,  # ty:ignore[invalid-argument-type]
                lstm_states.vf,  # ty:ignore[invalid-argument-type]
                episode_starts,
                self.lstm_critic,
            )
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)  # ty:ignore[call-non-callable]
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)  # ty:ignore[invalid-argument-type]

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[Distribution, tuple[torch.Tensor, ...]]:  # ty:ignore[invalid-method-override]
        """Get the current policy distribution given the observations.

        Args:
            obs: Observation.
            lstm_states: The last hidden and memory states for the LSTM.
            episode_starts: Whether the observations correspond to new episodes
                or not (we reset the lstm states in that case).

        Returns:
            the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)  # ty:ignore[invalid-assignment]
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:  # ty:ignore[invalid-method-override]
        """Get the estimated values according to the current policy given the observations.

        Args:
            obs: Observation.
            lstm_states: The last hidden and memory states for the LSTM.
            episode_starts: Whether the observations correspond to new episodes
                or not (we reset the lstm states in that case).

        Returns:
            the estimated values.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)  # ty:ignore[call-non-callable]

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, lstm_states: RNNStates, episode_starts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # ty:ignore[invalid-method-override]
        """Evaluate actions according to the current policy,.

        given the observations.

        Args:
            obs: Observation.
            actions:
            lstm_states: The last hidden and memory states for the LSTM.
            episode_starts: Whether the observations correspond to new episodes
                or not (we reset the lstm states in that case).

            actions: Actions taken.

        Returns:
            estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)  # ty:ignore[invalid-argument-type]
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)  # ty:ignore[invalid-argument-type]
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)  # ty:ignore[call-non-callable]

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()  # ty:ignore[invalid-return-type]

    def _predict(
        self,
        observation: torch.Tensor,
        lstm_states: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:  # ty:ignore[invalid-method-override]
        """Get the action according to the policy for a given observation.

        Args:
            observation:
            lstm_states: The last hidden and memory states for the LSTM.
            episode_starts: Whether the observations correspond to new episodes
                or not (we reset the lstm states in that case).
            deterministic: Whether to use stochastic or deterministic actions

            observation: The current observation.

        Returns:
            Taken action according to the policy and hidden states of the RNN
        """
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)  # ty:ignore[invalid-assignment]
        return distribution.get_actions(deterministic=deterministic), lstm_states

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
            lstm_states: The last hidden and memory states for the LSTM.
            episode_starts: Whether the observations correspond to new episodes
                or not (we reset the lstm states in that case).
            deterministic: Whether or not to return deterministic actions.
            state: The last hidden states for the LSTM.
            episode_start: Whether the episode is starting (triggers lstm state reset).

        Returns:
            the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)  # ty:ignore[invalid-assignment]

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]  # ty:ignore[invalid-argument-type]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)  # ty:ignore[invalid-assignment]
            state = (state, state)  # ty:ignore[invalid-assignment]

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with torch.no_grad():
            # Convert to PyTorch tensors
            states = (
                torch.tensor(state[0], dtype=torch.float32, device=self.device),  # ty:ignore[not-subscriptable]
                torch.tensor(state[1], dtype=torch.float32, device=self.device),  # ty:ignore[not-subscriptable]
            )
            episode_starts = torch.tensor(episode_start, dtype=torch.float32, device=self.device)
            actions, states = self._predict(
                observation,  # ty:ignore[invalid-argument-type]
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()

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
            actions = actions.squeeze(axis=0)

        return actions, states
