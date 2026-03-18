from collections.abc import Generator

from gymnasium import spaces
import numpy as np
import torch

from lit_sb3.buffers.rollout import RolloutBuffer
from lit_sb3.core.types import DictRolloutBufferSamples
from lit_sb3.vec_env import VecNormalize


class DictRolloutBuffer(RolloutBuffer):
    """Dict Rollout buffer used in on-policy algorithms like A2C/PPO.

    Extends the RolloutBuffer to use dictionary observations.

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    Args:
        buffer_size: Max number of element in the buffer
        observation_space: Observation space
        action_space: Action space
        device: PyTorch device
        gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Equivalent to Monte-Carlo advantage estimate when set to 1.
        gamma: Discount factor
        n_envs: Number of parallel environments
    """

    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]
    observations: dict[str, np.ndarray]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ) -> None:
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros(
                (self.buffer_size, self.n_envs, *obs_input_shape), dtype=self.observation_space[key].dtype
            )
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """Args.

        obs: Observation
        action: Action
        reward:
        episode_start: Start of episode signal.
        value: estimated value of the current state
            following the current policy.
        log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations:
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: int | None = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> DictRolloutBufferSamples:
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            actions=self.to_torch(self.actions[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )
