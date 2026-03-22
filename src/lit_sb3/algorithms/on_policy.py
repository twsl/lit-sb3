"""On-policy algorithm base class — Lightning-integrated.

One Lightning *epoch* corresponds to one rollout collection cycle:
  1. ``on_train_epoch_start``  → ``collect_rollouts()`` fills the RolloutBuffer.
  2. ``train_dataloader``      → wraps the buffer in a DataLoader (via RolloutDataset).
  3. ``training_step``         → per mini-batch gradient update (implemented by subclasses).
  4. ``configure_optimizers``  → returns the policy optimizer.

``learn()`` translates ``total_timesteps`` into Lightning ``max_epochs`` and
calls ``Trainer.fit(self)``, so the public SB3 API is fully preserved.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Optional, TypeVar, Union
import warnings

from gymnasium import spaces
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import numpy as np
import torch
from torch.utils.data import DataLoader

from lit_sb3.algorithms.base import BaseAlgorithm
from lit_sb3.buffers import DictRolloutBuffer, RolloutBuffer
from lit_sb3.callbacks import BaseCallback
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.data import RolloutDataset
from lit_sb3.policies import ActorCriticPolicy
from lit_sb3.utils.math import safe_mean
from lit_sb3.utils.tensors import obs_as_tensor
from lit_sb3.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """Base class for on-policy algorithms (PPO, A2C, TRPO, …).

    Subclasses must override ``training_step`` to compute and return the loss
    for a single mini-batch of rollout data.

    Args:
        policy: Policy model (MlpPolicy, CnnPolicy, …)
        env: Training environment
        learning_rate: LR or LR schedule (progress 1→0)
        n_steps: Rollout steps per env per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        ent_coef: Entropy coefficient
        vf_coef: Value-function coefficient
        max_grad_norm: Gradient clipping threshold
        use_sde: Use gSDE exploration
        sde_sample_freq: Noise re-sample frequency for gSDE
        rollout_buffer_class: Buffer class (auto-selected if None)
        rollout_buffer_kwargs: Extra kwargs for the buffer constructor
        stats_window_size: Episode-info averaging window
        monitor_wrapper: Wrap env in Monitor
        policy_kwargs: Extra kwargs passed to the policy
        verbose: Verbosity level
        seed: RNG seed
        device: Torch device
        _init_setup_model: Call ``_setup_model()`` in ``__init__``
        supported_action_spaces: Allowed action-space types
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        learning_rate: float | Schedule,
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        monitor_wrapper: bool = True,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: tuple[type[spaces.Space], ...] | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            supported_action_spaces=supported_action_spaces,
        )
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if _init_setup_model and hasattr(self, "observation_space"):
            self._setup_model()

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        policy_class_name = self.policy_class.__name__
        if self.device != torch.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Lightning hooks — training loop
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the policy optimizer for Lightning's automatic optimization."""
        return self.policy.optimizer

    def train_dataloader(self) -> DataLoader:
        """Wrap the rollout buffer in a DataLoader.

        ``RolloutDataset`` holds a *reference* to the buffer, so the freshly
        collected rollout data (stored in ``on_train_epoch_start``) is
        automatically served to each ``training_step`` call.
        """
        dataset = RolloutDataset(
            self.rollout_buffer,
            batch_size=self._rl_batch_size,  # ty:ignore[invalid-argument-type]
            n_epochs=self._rl_n_epochs,
        )
        return DataLoader(dataset, batch_size=None, collate_fn=lambda x: x, num_workers=0)

    def on_train_epoch_start(self) -> None:
        """Collect a fresh rollout (unless skipped) and update progress/LR before each epoch."""
        assert self.env is not None, "Environment must be set before training."
        assert hasattr(self, "_rl_callback"), "Call learn() to start training."

        if getattr(self, "_skip_next_collection", False):
            # First epoch: rollout was already pre-collected in learn()
            self._skip_next_collection = False
        else:
            self._rollout_ok = self.collect_rollouts(
                self.env,
                self._rl_callback,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps,
            )

        if not getattr(self, "_rollout_ok", True):
            self.trainer.should_stop = True
            return

        self._rl_iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
        self._update_learning_rate(self.policy.optimizer)

        # Log rollout stats
        if self.ep_info_buffer is not None and len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            ep_rew = float(safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            ep_len = float(safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.log("rollout/ep_rew_mean", ep_rew, on_step=False, on_epoch=True, prog_bar=True)
            self.log("rollout/ep_len_mean", ep_len, on_step=False, on_epoch=True, prog_bar=False)
        if self.ep_success_buffer is not None and len(self.ep_success_buffer) > 0:
            self.log(
                "rollout/success_rate",
                float(safe_mean(self.ep_success_buffer)),
                on_step=False,
                on_epoch=True,
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute and return the loss for one mini-batch of rollout data.

        Must be overridden by concrete algorithm classes (PPO, A2C, …).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `training_step`.")

    def on_train_end(self) -> None:
        """Signal the RL callback that training has finished."""
        if hasattr(self, "_rl_callback"):
            self._rl_callback.on_training_end()

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect experiences using the current policy and fill ``rollout_buffer``.

        Args:
            env: Training environment
            callback: Callback called at each step
            rollout_buffer: Buffer to fill
            n_rollout_steps: Number of steps per env

        Returns:
            True if completed, False if callback terminated early
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            budget_reached = self._has_reached_rl_budget()

            callback.update_locals(locals())
            continue_training = callback.on_step()
            if not continue_training and not budget_reached:
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # RLTrainer protocol
    # ------------------------------------------------------------------

    @property
    def _default_log_every_n_steps(self) -> int:
        return 1

    def _prepare_fit(
        self,
        callback: BaseCallback,
        log_every_n_steps: int,
    ) -> None:
        """Inject on-policy state needed by Lightning hooks."""
        self._rl_callback = callback
        self._rl_iteration = 0
        self._log_interval = log_every_n_steps
        # Allow subclasses to set batch_size/n_epochs in __init__; fall back to defaults.
        if not hasattr(self, "_rl_batch_size"):
            self._rl_batch_size = None
        if not hasattr(self, "_rl_n_epochs"):
            self._rl_n_epochs = 1

    def _rl_max_epochs(self, total_timesteps: int) -> int:
        return max(1, math.ceil(total_timesteps / (self.n_steps * self.n_envs)))

    def _pre_fit_hook(self, callback: BaseCallback) -> None:
        """Pre-collect the first rollout so Lightning's setup_data() prefetch finds a fully populated rollout buffer."""
        self.collect_rollouts(
            self.env,  # ty:ignore[invalid-argument-type]
            callback,
            self.rollout_buffer,
            n_rollout_steps=self.n_steps,
        )
        self._skip_next_collection = True  # on_train_epoch_start skips for epoch 1

    # ------------------------------------------------------------------
    # SB3-compatible learn() — delegates to RLTrainer
    # ------------------------------------------------------------------

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        env: GymEnv | str | None = None,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = None,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        default_root_dir: str | Path | None = None,
    ) -> SelfOnPolicyAlgorithm:
        """Train for ``total_timesteps`` environment steps.

        Args:
            total_timesteps: Total environment steps to collect.
            env: Training environment (gym env, env-id string, or VecEnv).
            callback: SB3-style callback(s) fired at each RL step.
            log_every_n_steps: Logging frequency.
            tb_log_name: TensorBoard run name.
            reset_num_timesteps: Whether to reset the step counter.
            progress_bar: Display a progress bar.
            default_root_dir: Root directory for logs and checkpoints.

        Returns:
            The trained model.
        """
        from lit_sb3.datamodule import RLEnvDataModule

        if env is None:
            raise TypeError("learn() requires an env argument (environment removed from model constructor).")

        dm = RLEnvDataModule(train_env=env, verbose=self.verbose)
        trainer = self.create_trainer(
            max_steps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            run_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            enable_progress_bar=progress_bar,
            default_root_dir=default_root_dir,
        )
        return trainer.fit(self, datamodule=dm)  # type: ignore[return-value]

    def dump_logs(self, iteration: int = 0) -> None:
        """Write log data (kept for SB3 API compatibility)."""
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        metrics: dict[str, Any] = {
            "time/fps": fps,
        }
        if iteration > 0:
            metrics["time/iterations"] = iteration
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            metrics["rollout/ep_rew_mean"] = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            metrics["rollout/ep_len_mean"] = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
        if len(self.ep_success_buffer) > 0:
            metrics["rollout/success_rate"] = safe_mean(self.ep_success_buffer)
        self.log_metrics(metrics, step=self.num_timesteps)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
