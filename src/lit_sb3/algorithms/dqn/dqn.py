from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union
import warnings

from gymnasium import spaces
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from lit_sb3.algorithms.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from lit_sb3.algorithms.off_policy import OffPolicyAlgorithm
from lit_sb3.buffers import ReplayBuffer
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.policies import BasePolicy
from lit_sb3.utils.schedule import LinearSchedule
from lit_sb3.utils.tensors import get_parameters_by_name, polyak_update

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class DQN(OffPolicyAlgorithm):
    """Deep Q-Network (DQN).

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from (if registered in Gym, can be str)
        learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        buffer_size: size of the replay buffer
        learning_starts: how many steps of the model to collect transitions for before learning starts
        batch_size: Minibatch size for each gradient update
        tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
        gamma: the discount factor
        train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
            like ``(5, "step")`` or ``(2, "episode")``.
        gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout.
        replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
            If ``None``, it will be automatically selected.
        replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
        optimize_memory_usage: Enable a memory efficient variant of the replay buffer
            at a cost of more complexity.
            See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
        target_update_interval: update the target network every ``target_update_interval``
            environment steps.
        exploration_fraction: fraction of entire training period over which the exploration rate is reduced
        exploration_initial_eps: initial value of random action probability
        exploration_final_eps: final value of random action probability
        max_grad_norm: The maximum value for the gradient clipping
        stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        tensorboard_log: the log location for tensorboard (if None, no logging)
        policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`dqn_policies`
        verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        seed: Seed for the pseudo random generators
        device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: str | type[DQNPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: str | Path | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1 and self.n_envs > self.target_update_interval:
            warnings.warn(
                "The number of environments used is greater than the target network "
                f"update interval ({self.n_envs} > {self.target_update_interval}), "
                "therefore the target network will be updated after each call to env.step() "
                f"which corresponds to {self.n_envs} steps.",
                stacklevel=2,
            )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """Update the exploration rate and target network if needed.

        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    # ------------------------------------------------------------------
    # Lightning hooks (manual optimization)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.policy.optimizer

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """Off-policy DQN step: collect env transitions then do gradient updates."""
        assert self.env is not None

        rollout = self.collect_rollouts(
            self.env,
            train_freq=self.train_freq,  # ty:ignore[invalid-argument-type]
            action_noise=self.action_noise,
            callback=self._rl_callback,
            learning_starts=self.learning_starts,
            replay_buffer=self.replay_buffer,  # ty:ignore[invalid-argument-type]
            log_interval=self._log_interval,
        )

        if not rollout.continue_training:
            self.trainer.should_stop = True
            return

        if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
            gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
            if gradient_steps > 0:
                stats = self._do_gradient_steps(gradient_steps, self.batch_size)
                self.log("train/n_updates", float(self._n_updates), on_step=True, on_epoch=False)
                self.log("train/loss", stats["loss"], on_step=True, on_epoch=False)
                return

    def _do_gradient_steps(self, gradient_steps: int, batch_size: int) -> dict[str, float]:
        """DQN gradient updates (manual optimization).

        Returns a dict of scalar stats (losses) for the caller to log.
        """
        opt = self.optimizers()

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            opt.zero_grad()  # ty:ignore[unresolved-attribute]
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            opt.step()  # ty:ignore[unresolved-attribute]

        self._n_updates += gradient_steps
        return {"loss": float(np.mean(losses))}

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Overrides the base_class predict function to include epsilon-greedy exploration.

        Args:
            observation: the input observation
            state: The last states (can be None, used in recurrent policies)
            episode_start: The last masks (can be None, used in recurrent policies)
            deterministic: Whether or not to return deterministic actions.

        Returns:
            the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]  # ty:ignore[invalid-argument-type]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
