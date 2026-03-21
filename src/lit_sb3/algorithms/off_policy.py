"""Off-policy algorithm base class — Lightning-integrated.

Architecture:
  * ``automatic_optimization = False`` — all gradient calls are explicit.
  * ``train_dataloader`` returns a ``ReplayBufferDataset`` that drives one
    training step per ``total_timesteps`` index.
  * ``training_step`` (overridden by subclasses) collects env transitions
    and performs manual gradient updates.
  * ``learn()`` wraps Lightning ``Trainer.fit`` while keeping the SB3 API.
"""

from __future__ import annotations

from copy import deepcopy
import io
import pathlib
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
from lit_sb3.algorithms.her.her_replay_buffer import HerReplayBuffer
from lit_sb3.buffers.dict_replay import DictReplayBuffer
from lit_sb3.buffers.nstep_replay import NStepReplayBuffer
from lit_sb3.buffers.replay import ReplayBuffer
from lit_sb3.callbacks import BaseCallback
from lit_sb3.core.noise import ActionNoise, VectorizedActionNoise
from lit_sb3.core.types import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from lit_sb3.data.replay import ReplayBufferDataset
from lit_sb3.policies import BasePolicy
from lit_sb3.utils.math import safe_mean
from lit_sb3.utils.save import load_from_pkl, save_to_pkl
from lit_sb3.utils.training import should_collect_more_steps
from lit_sb3.vec_env import VecEnv

SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")


class OffPolicyAlgorithm(BaseAlgorithm):
    """Base class for off-policy algorithms (SAC, TD3, DQN, DDPG, …).

    Off-policy training is driven step-by-step:
    each Lightning ``training_step`` call represents one iteration of
    "collect some env experience → do gradient updates from replay buffer".

    Subclasses must implement ``training_step`` and ``configure_optimizers``.

    Args:
        policy: Policy model
        env: Training environment
        learning_rate: LR or LR schedule
        buffer_size: Replay buffer capacity
        learning_starts: Warm-up steps before gradient updates begin
        batch_size: Mini-batch size for gradient updates
        tau: Polyak soft-update coefficient
        gamma: Discount factor
        train_freq: How often to collect env steps (per-step or per-episode)
        gradient_steps: Gradient updates after each collection; ``-1`` = auto
        action_noise: Exploration noise (for deterministic policies)
        replay_buffer_class: Buffer class (auto-selected if None)
        replay_buffer_kwargs: Extra kwargs for the buffer
        optimize_memory_usage: Memory-efficient replay buffer variant
        n_steps: N-step returns (>1 uses NStepReplayBuffer)
        policy_kwargs: Extra kwargs passed to the policy
        stats_window_size: Episode-info averaging window
        tensorboard_log: TensorBoard log directory
        verbose: Verbosity level
        device: Torch device
        support_multi_env: Allow multiple parallel envs
        monitor_wrapper: Wrap env in Monitor
        seed: RNG seed
        use_sde: Use gSDE exploration
        sde_sample_freq: Noise re-sample frequency for gSDE
        use_sde_at_warmup: Use gSDE during warm-up phase
        sde_support: Whether the policy supports gSDE
        supported_action_spaces: Allowed action-space types
    """

    # Disable Lightning's automatic optimizer stepping — we call step/backward manually
    automatic_optimization = False

    actor: torch.nn.Module

    def __init__(
        self,
        policy: str | type[BasePolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | Path | None = None,
        verbose: int = 0,
        device: torch.device | str = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: int | None = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: tuple[type[spaces.Space], ...] | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer: ReplayBuffer | None = None
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs or {}
        self.n_steps = n_steps
        self.train_freq = train_freq  # converted in _setup_model via _convert_train_freq

        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        self.use_sde_at_warmup = use_sde_at_warmup

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _convert_train_freq(self) -> None:
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")
            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError as e:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
                ) from e
            if not isinstance(train_freq[0], int):
                raise TypeError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")
            self.train_freq = TrainFreq(*train_freq)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
                assert self.n_steps == 1, "N-step returns are not supported for Dict observation spaces yet."
            elif self.n_steps > 1:
                self.replay_buffer_class = NStepReplayBuffer
                self.replay_buffer_kwargs.update({"n_steps": self.n_steps, "gamma": self.gamma})
            else:
                self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self._convert_train_freq()

    # ------------------------------------------------------------------
    # Lightning hooks — training loop
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Subclasses must override this to return their optimizers."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement `configure_optimizers`.")

    def train_dataloader(self) -> DataLoader:
        """Return an infinite-like dataset that drives the training loop.

        Each integer token yielded by ``ReplayBufferDataset`` triggers one
        call to ``training_step``, which handles env interaction + gradient updates.
        """
        dataset = ReplayBufferDataset(self._total_timesteps)
        return DataLoader(dataset, batch_size=1, num_workers=0)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """One off-policy iteration: collect env transitions + gradient updates.

        Must be overridden by concrete algorithm classes.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `training_step`.")

    def on_train_end(self) -> None:
        if hasattr(self, "_rl_callback"):
            self._rl_callback.on_training_end()

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            assert self._last_obs is not None
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)  # ty:ignore[invalid-argument-type]

        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray | dict[str, np.ndarray],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        next_obs = deepcopy(new_obs_)
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs:
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
        )
        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: ActionNoise | None = None,
        learning_starts: int = 0,
        log_interval: int | None = None,
    ) -> RolloutReturn:
        self.policy.set_training_mode(False)
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv)
        assert train_freq.frequency > 0
        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            self._update_info_buffer(infos, dones)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1
                    if action_noise is not None:
                        kwargs = {"indices": [idx]} if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()

        callback.on_rollout_end()
        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _on_step(self) -> None:
        """Called after each env step; override for per-step hooks (e.g. DQN target update)."""
        pass

    def dump_logs(self) -> None:
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())  # type: ignore[operator]
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    # ------------------------------------------------------------------
    # Replay buffer I/O
    # ------------------------------------------------------------------

    def save_replay_buffer(self, path: str | pathlib.Path | io.BufferedIOBase) -> None:
        assert self.replay_buffer is not None
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: str | pathlib.Path | io.BufferedIOBase,
        truncate_last_traj: bool = True,
    ) -> None:
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer)
        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)
        if isinstance(self.replay_buffer, HerReplayBuffer):
            assert self.env is not None
            self.replay_buffer.set_env(self.env)
            if truncate_last_traj:
                self.replay_buffer.truncate_last_trajectory()
        self.replay_buffer.device = self.device

    # ------------------------------------------------------------------
    # SB3-compatible _setup_learn / learn
    # ------------------------------------------------------------------

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        replay_buffer = self.replay_buffer
        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )
        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue.",
                stacklevel=2,
            )
            assert replay_buffer is not None
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        assert self.env is not None
        if (
            self.action_noise is not None
            and self.env.num_envs > 1
            and not isinstance(self.action_noise, VectorizedActionNoise)
        ):
            self.action_noise = VectorizedActionNoise(self.action_noise, self.env.num_envs)

        return super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)

    # ------------------------------------------------------------------
    # RLTrainer protocol
    # ------------------------------------------------------------------

    @property
    def _default_log_every_n_steps(self) -> int:
        return 4

    def _prepare_fit(
        self,
        callback: BaseCallback,
        log_every_n_steps: int,
    ) -> None:
        """Inject off-policy state needed by Lightning hooks."""
        assert isinstance(self.train_freq, TrainFreq)
        self._rl_callback = callback
        self._log_interval = log_every_n_steps

    def _rl_max_epochs(self, total_timesteps: int) -> int:
        # Off-policy uses a single epoch driven by ReplayBufferDataset length.
        return 1

    # _pre_fit_hook: no pre-collection needed for off-policy; default no-op is used.

    # ------------------------------------------------------------------
    # SB3-compatible learn() — delegates to RLTrainer
    # ------------------------------------------------------------------

    def learn(
        self: SelfOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = None,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        """Train for ``total_timesteps`` env steps.

        Delegates to :class:`~lit_sb3.trainer.RLTrainer`.  For full control
        over Lightning Trainer settings (precision, loggers, plugins …) construct
        ``RLTrainer`` directly::

            RLTrainer(max_steps=total_timesteps, ...).fit(model)
        """
        from lit_sb3.trainer import RLTrainer

        return RLTrainer(  # type: ignore[return-value]
            max_steps=total_timesteps,
            rl_callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            enable_progress_bar=progress_bar,
        ).fit(self)
