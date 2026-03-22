from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union

from gymnasium import spaces
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from lit_sb3.algorithms.off_policy import OffPolicyAlgorithm
from lit_sb3.algorithms.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from lit_sb3.buffers import ReplayBuffer
from lit_sb3.core.noise import ActionNoise
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.policies import BasePolicy, ContinuousCritic
from lit_sb3.utils.tensors import get_parameters_by_name, polyak_update

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class SAC(OffPolicyAlgorithm):
    """Soft Actor-Critic (SAC).

    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html.

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from (if registered in Gym, can be str)
        learning_rate: learning rate for adam optimizer,
            the same learning rate will be used for all networks (Q-Values, Actor and Value function)
            it can be a function of the current progress remaining (from 1 to 0)
        buffer_size: size of the replay buffer
        learning_starts: how many steps of the model to collect transitions for before learning starts
        batch_size: Minibatch size for each gradient update
        tau: the soft update coefficient ("Polyak update", between 0 and 1)
        gamma: the discount factor
        train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
            like ``(5, "step")`` or ``(2, "episode")``.
        gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout.
        action_noise: the action noise type (None by default), this can help
            for hard exploration problem. Cf common.noise for the different action noise type.
        replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
            If ``None``, it will be automatically selected.
        replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
        optimize_memory_usage: Enable a memory efficient variant of the replay buffer
            at a cost of more complexity.
            See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
        ent_coef: Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
            Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
        target_update_interval: update the target network every ``target_network_update_freq``
            gradient steps.
        target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
        use_sde: Whether to use generalized State Dependent Exploration (gSDE)
            instead of action noise exploration (default: False)
        sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
            during the warm up phase (before learning starts)
        stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        tensorboard_log: the log location for tensorboard (if None, no logging)
        policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
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
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: str | type[SACPolicy],
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
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
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
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
        self.log_ent_coef = None  # type: Optional[torch.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: torch.optim.Adam | None = None

        if _init_setup_model and hasattr(self, "observation_space"):
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.action_space.shape).astype(np.float32))  # type: ignore[union-attr]
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
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
        """Off-policy SAC step: collect env transitions then do gradient updates."""
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
        """SAC gradient updates (manual optimization).

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

            # --- Entropy coefficient update ---
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

            # --- Critic update ---
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q, _ = torch.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q

            current_q = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())
            critic_opt.zero_grad()
            self.manual_backward(critic_loss)
            critic_opt.step()

            # --- Actor update ---
            q_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_q_pi, _ = torch.min(q_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()
            actor_losses.append(actor_loss.item())
            actor_opt.zero_grad()
            self.manual_backward(actor_loss)
            actor_opt.step()

            # --- Target network soft update ---
            if gradient_step % self.target_update_interval == 0:
                from lit_sb3.utils.tensors import polyak_update

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
        self: SelfSAC,
        total_timesteps: int,
        env: GymEnv | str | None = None,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        default_root_dir: str | Path | None = None,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            env=env,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            default_root_dir=default_root_dir,
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
