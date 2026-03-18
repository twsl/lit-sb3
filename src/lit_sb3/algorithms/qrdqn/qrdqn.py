from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union
import warnings

from gymnasium import spaces
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import numpy as np
import torch

from lit_sb3.algorithms.off_policy import OffPolicyAlgorithm
from lit_sb3.algorithms.qrdqn.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, QRDQNPolicy, QuantileNetwork
from lit_sb3.buffers import ReplayBuffer
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.core.utils import LinearSchedule, get_parameters_by_name, polyak_update, quantile_huber_loss
from lit_sb3.policies import BasePolicy

SelfQRDQN = TypeVar("SelfQRDQN", bound="QRDQN")


class QRDQN(OffPolicyAlgorithm):
    """Quantile Regression Deep Q-Network (QR-DQN).

    Paper: https://arxiv.org/abs/1710.10044.

    Adapted from sb3-contrib to use PyTorch Lightning.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    exploration_schedule: Schedule
    quantile_net: QuantileNetwork
    quantile_net_target: QuantileNetwork
    policy: QRDQNPolicy

    def __init__(
        self,
        policy: str | type[QRDQNPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 5e-5,
        buffer_size: int = 1_000_000,
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
        exploration_fraction: float = 0.005,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        max_grad_norm: float | None = None,
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
            action_noise=None,
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

        self.n_quantiles = None  # defined in policy
        self.target_update_interval = target_update_interval
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.quantile_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.quantile_net_target, ["running_"])
        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        if self.n_envs > 1 and self.n_envs > self.target_update_interval:
            warnings.warn(
                f"The number of environments used is greater than the target network "
                f"update interval ({self.n_envs} > {self.target_update_interval}), "
                "therefore the target network will be updated after each call to env.step().",
                stacklevel=2,
            )

    def _create_aliases(self) -> None:
        self.quantile_net = self.policy.quantile_net
        self.quantile_net_target = self.policy.quantile_net_target
        self.n_quantiles = self.policy.n_quantiles

    def _on_step(self) -> None:
        """Update the target network and exploration rate."""
        self._n_calls += 1
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.quantile_net.parameters(), self.quantile_net_target.parameters(), self.tau)
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)

    # ------------------------------------------------------------------
    # Lightning hooks (manual optimization)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.policy.optimizer

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """Off-policy QRDQN step: collect env transitions then do gradient updates."""
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
                self._do_gradient_steps(gradient_steps, self.batch_size)

    def _do_gradient_steps(self, gradient_steps: int, batch_size: int) -> None:
        """QRDQN gradient updates (manual optimization with quantile Huber loss)."""
        opt = self.optimizers()

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
                next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
                next_greedy_actions = next_greedy_actions.expand(batch_size, self.n_quantiles, 1)
                next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles

            current_quantiles = self.quantile_net(replay_data.observations)
            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)  # ty:ignore[invalid-argument-type]
            current_quantiles = torch.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

            loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
            losses.append(loss.item())

            opt.zero_grad()  # ty:ignore[unresolved-attribute]
            self.manual_backward(loss)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            opt.step()  # ty:ignore[unresolved-attribute]

        self._n_updates += gradient_steps
        self.log("train/n_updates", float(self._n_updates), on_step=True, on_epoch=False)
        self.log("train/loss", float(np.mean(losses)), on_step=True, on_epoch=False)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Epsilon-greedy action selection for QRDQN."""
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
        self: SelfQRDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 4,
        tb_log_name: str = "QRDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfQRDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["quantile_net", "quantile_net_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return ["policy", "policy.optimizer"], []
