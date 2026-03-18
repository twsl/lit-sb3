import numpy as np

from lit_sb3.buffers.replay import ReplayBuffer
from lit_sb3.core.types import ReplayBufferSamples
from lit_sb3.vec_env import VecNormalize


class NStepReplayBuffer(ReplayBuffer):
    """Replay buffer used for computing n-step returns in off-policy algorithms like SAC/DQN.

    The n-step return combines multiple steps of future rewards,
    discounted by the discount factor gamma.
    This can help improve sample efficiency and credit assignment.

    This implementation uses the same storage space as a normal replay buffer,
    and NumPy vectorized operations at sampling time to efficiently compute the
    n-step return, without requiring extra memory.

    This implementation is inspired by:
    - https://github.com/younggyoseo/FastTD3
    - https://github.com/DLR-RM/stable-baselines3/pull/81

    It avoids potential issues such as:
    - https://github.com/younggyoseo/FastTD3/issues/6

    Args:
        buffer_size: Max number of element in the buffer
        observation_space: Observation space
        action_space: Action space
        device: PyTorch device
        n_envs: Number of parallel environments
        optimize_memory_usage: Not supported
        handle_timeout_termination: Handle timeout termination (due to timelimit)
            separately and treat the task as infinite horizon task.
            https://github.com/DLR-RM/stable-baselines3/issues/284
        n_steps: Number of steps to accumulate rewards for n-step returns
        gamma: Discount factor for future rewards
    """

    def __init__(self, *args, n_steps: int = 3, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma
        if self.optimize_memory_usage:
            raise NotImplementedError("NStepReplayBuffer doesn't support optimize_memory_usage=True")

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> ReplayBufferSamples:
        """Sample a batch of transitions and compute n-step returns.

        For each sampled transition, the method computes the cumulative discounted reward over
        the next `n_steps`, properly handling episode termination and timeouts.
        The next observation and done flag correspond to the last transition in the computed n-step trajectory.

        Args:
            batch_inds: Indices of samples to retrieve
            env: Optional VecNormalize environment for normalizing observations/rewards

        Returns:
            A batch of samples with n-step returns and corresponding observations/actions
        """
        # Randomly choose env indices for each sample
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)

        # Note: the self.pos index is dangerous (will overlap two different episodes when buffer is full)
        # so we set self.pos-1 to truncated=True (temporarily) if done=False and truncated=False
        last_valid_index = self.pos - 1
        original_timeout_values = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(
            original_timeout_values, np.logical_not(self.dones[last_valid_index])
        )

        # Compute n-step indices with wrap-around
        steps = np.arange(self.n_steps).reshape(1, -1)  # shape: [1, n_steps]
        indices = (batch_inds[:, None] + steps) % self.buffer_size  # shape: [batch, n_steps]

        # Retrieve sequences of transitions
        rewards_seq = self._normalize_reward(self.rewards[indices, env_indices[:, None]], env)  # [batch, n_steps]
        dones_seq = self.dones[indices, env_indices[:, None]]  # [batch, n_steps]
        truncated_seq = self.timeouts[indices, env_indices[:, None]]  # [batch, n_steps]

        # Compute masks: 1 until first done/truncation (inclusive)
        done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        done_idx = done_or_truncated.argmax(axis=1)
        # If no done/truncation, keep full sequence
        has_done_or_truncated = done_or_truncated.any(axis=1)
        done_idx = np.where(has_done_or_truncated, done_idx, self.n_steps - 1)

        mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]  # shape: [batch, n_steps]
        # Compute discount factors for bootstrapping (using target Q-Value)
        # It is gamma ** n_steps by default but should be adjusted in case of early termination/truncation.
        target_q_discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)  # [batch, 1]

        # Apply discount
        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(1, -1)  # [1, n_steps]
        discounted_rewards = rewards_seq * discounts * mask
        n_step_returns = discounted_rewards.sum(axis=1, keepdims=True)  # [batch, 1]

        # Compute indices of next_obs/done at the final point of the n-step transition
        last_indices = (batch_inds + done_idx) % self.buffer_size
        next_obs = self._normalize_obs(self.next_observations[last_indices, env_indices], env)
        next_dones = self.dones[last_indices, env_indices][:, None].astype(np.float32)
        next_timeouts = self.timeouts[last_indices, env_indices][:, None].astype(np.float32)
        final_dones = next_dones * (1.0 - next_timeouts)

        # Revert back tmp changes to avoid sampling across episodes
        self.timeouts[last_valid_index] = original_timeout_values

        # Gather observations and actions
        obs = self._normalize_obs(self.observations[batch_inds, env_indices], env)
        actions = self.actions[batch_inds, env_indices]

        return ReplayBufferSamples(
            observations=self.to_torch(obs),  # type: ignore[arg-type]
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),  # type: ignore[arg-type]
            dones=self.to_torch(final_dones),
            rewards=self.to_torch(n_step_returns),
            discounts=self.to_torch(target_q_discounts),
        )
