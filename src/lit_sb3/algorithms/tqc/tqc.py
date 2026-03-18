from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union

from gymnasium import spaces
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import numpy as np
import torch

from lit_sb3.algorithms.off_policy import OffPolicyAlgorithm
from lit_sb3.algorithms.tqc.policies import Actor, CnnPolicy, Critic, MlpPolicy, MultiInputPolicy, TQCPolicy
from lit_sb3.buffers import ReplayBuffer
from lit_sb3.core.noise import ActionNoise
from lit_sb3.core.types import GymEnv, MaybeCallback
from lit_sb3.core.utils import get_parameters_by_name, polyak_update, quantile_huber_loss
from lit_sb3.policies import BasePolicy

SelfTQC = TypeVar("SelfTQC", bound="TQC")


class TQC(OffPolicyAlgorithm):
    """Controlling Overestimation Bias with Truncated Mixture of Continuous.

    Distributional Quantile Critics (TQC).

    Paper: https://arxiv.org/abs/2005.04269
    This implementation adapts TQC from sb3-contrib to use PyTorch Lightning.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TQCPolicy
    actor: Actor
    critic: Critic
    critic_target: Critic

    def __init__(
        self,
        policy: str | type[TQCPolicy],
        env: GymEnv | str,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
        top_quantiles_to_drop_per_net: int = 2,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
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
            action_noise=action_noise,
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
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef: torch.Tensor | None = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: torch.optim.Adam | None = None
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore[union-attr]
        else:
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    # ------------------------------------------------------------------
    # Lightning hooks (manual optimization)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        return optimizers

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """Off-policy TQC step: collect env transitions then do gradient updates."""
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
                self.log_dict(
                    {
                        "train/n_updates": float(self._n_updates),
                        "train/ent_coef": stats["ent_coef"],
                        "train/actor_loss": stats["actor_loss"],
                        "train/critic_loss": stats["critic_loss"],
                    },
                    on_step=True,
                    on_epoch=False,
                )
                if "ent_coef_loss" in stats:
                    self.log("train/ent_coef_loss", stats["ent_coef_loss"], on_step=True, on_epoch=False)

    def _do_gradient_steps(self, gradient_steps: int, batch_size: int) -> dict[str, float]:
        """TQC gradient updates (manual optimization with quantile critic).

        Returns a dict of scalar stats (losses, ent_coef) for the caller to log.
        """
        optimizers = self.optimizers()
        actor_opt = optimizers[0]  # ty:ignore[not-subscriptable]
        critic_opt = optimizers[1]  # ty:ignore[not-subscriptable]
        ent_opt = optimizers[2] if len(optimizers) > 2 else None  # ty:ignore[invalid-argument-type, not-subscriptable]

        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [self.actor.optimizer, self.critic.optimizer]
            + ([self.ent_coef_optimizer] if self.ent_coef_optimizer is not None else [])
        )

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Entropy coef update
            if ent_opt is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
                ent_opt.zero_grad()
                self.manual_backward(ent_coef_loss)
                ent_opt.step()
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                n_target_quantiles = (
                    self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                )
                next_quantiles, _ = torch.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target_quantiles]
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * target_quantiles
                target_quantiles.unsqueeze_(dim=1)

            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())
            critic_opt.zero_grad()
            self.manual_backward(critic_loss)
            critic_opt.step()

            # Actor update
            qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - qf_pi).mean()
            actor_losses.append(actor_loss.item())
            actor_opt.zero_grad()
            self.manual_backward(actor_loss)
            actor_opt.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        stats: dict[str, float] = {
            "ent_coef": float(np.mean(ent_coefs)),
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
        }
        if ent_coef_losses:
            stats["ent_coef_loss"] = float(np.mean(ent_coef_losses))
        return stats

    def learn(
        self: SelfTQC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 4,
        tb_log_name: str = "TQC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTQC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
