from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union
import warnings

from gymnasium import spaces
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from lit_sb3.algorithms.on_policy import OnPolicyAlgorithm
from lit_sb3.buffers import RolloutBuffer
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.core.utils import FloatSchedule, explained_variance
from lit_sb3.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """Proximal Policy Optimization algorithm (PPO) (clip version).

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from (if registered in Gym, can be str)
        learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        n_steps: The number of steps to run for each environment per update
            (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in
            parallel)
            NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
            See https://github.com/pytorch/pytorch/issues/29372
        batch_size: Minibatch size
        n_epochs: Number of epoch when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range: Clipping parameter, it can be a function of the current progress
            remaining (from 1 to 0).
        clip_range_vf: Clipping parameter for the value function,
            it can be a function of the current progress remaining (from 1 to 0).
            This is a parameter specific to the OpenAI implementation. If None is passed (default),
            no clipping will be done on the value function.
            IMPORTANT: this clipping depends on the reward scaling.
        normalize_advantage: Whether to normalize or not the advantage
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: The maximum value for the gradient clipping
        use_sde: Whether to use generalized State Dependent Exploration (gSDE)
            instead of action noise exploration (default: False)
        sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
        rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
        target_kl: Limit the KL divergence between updates,
            because the clipping is not enough to prevent large update
            see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
            By default, there is no limit on the kl div.
        stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        tensorboard_log: the log location for tensorboard (if None, no logging)
        policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
        verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        seed: Seed for the pseudo random generators
        device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        target_kl: float | None = None,
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
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert batch_size > 1, (
                "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
            )

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (not normalize_advantage), (
                f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            )
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})",
                    stacklevel=2,
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        # Lightning DataLoader config (must be set before train_dataloader() is called)
        self._rl_batch_size = batch_size
        self._rl_n_epochs = n_epochs

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the policy optimizer."""
        return self.policy.optimizer

    def on_train_epoch_start(self) -> None:
        """Run base rollout collection hook then set policy to train mode."""
        super().on_train_epoch_start()
        self.policy.set_training_mode(True)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute PPO clipped-surrogate loss for one mini-batch of rollout data."""
        rollout_data = batch

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = rollout_data.actions.long().flatten()

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        ratio = torch.exp(log_prob - rollout_data.old_log_prob)
        policy_loss = -torch.min(
            advantages * ratio, advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        ).mean()

        if self.clip_range_vf is None:
            values_pred = values
        else:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
            values_pred = rollout_data.old_values + torch.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        value_loss = F.mse_loss(rollout_data.returns, values_pred)

        entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # KL divergence check — signal early stopping to Lightning
        with torch.no_grad():
            log_ratio = log_prob - rollout_data.old_log_prob
            approx_kl_div = float(torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().item())

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
            if self.verbose >= 1:
                print(f"Early stopping due to reaching max kl: {approx_kl_div:.2f}")
            self.trainer.should_stop = True

        self._n_updates += 1
        self.log_dict(
            {
                "train/policy_loss": policy_loss.item(),
                "train/value_loss": value_loss.item(),
                "train/entropy_loss": entropy_loss.item(),
                "train/approx_kl": approx_kl_div,
                "train/clip_range": clip_range,
                "train/loss": loss.item(),
            },
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        if hasattr(self.policy, "log_std"):
            self.log("train/std", torch.exp(self.policy.log_std).mean().item(), on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        """Log epoch-level PPO stats."""
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.log("train/explained_variance", float(explained_var), on_step=False, on_epoch=True)
        self.log("train/n_updates", float(self._n_updates), on_step=False, on_epoch=True)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
