from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union

from gymnasium import spaces
import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from lit_sb3.algorithms.crossq.policies import Actor, CrossQCritic, CrossQPolicy, MlpPolicy
from lit_sb3.algorithms.off_policy import OffPolicyAlgorithm
from lit_sb3.buffers import ReplayBuffer
from lit_sb3.core.noise import ActionNoise
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.policies import BasePolicy

SelfCrossQ = TypeVar("SelfCrossQ", bound="CrossQ")


class CrossQ(OffPolicyAlgorithm):
    """Implementation of Batch Normalization in Deep Reinforcement Learning (CrossQ).

    Paper: https://openreview.net/forum?id=PczQtTsTIX
    Reference implementation: https://github.com/araffin/sbx.

    Args:
        policy: The policy model to use (MlpPolicy)
        env: The environment to learn from (if registered in Gym, can be str)
        learning_rate: learning rate for adam optimizer,
            the same learning rate will be used for all networks (Q-Values, Actor and Value function)
            it can be a function of the current progress remaining (from 1 to 0)
        buffer_size: size of the replay buffer
        learning_starts: how many steps of the model to collect transitions for before learning starts
        batch_size: Minibatch size for each gradient update
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
        policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`crossq_policies`
        verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        seed: Seed for the pseudo random generators
        device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        # TODO: Implement CnnPolicy and MultiInputPolicy
    }
    policy: CrossQPolicy
    actor: Actor
    critic: CrossQCritic

    def __init__(
        self,
        policy: str | type[CrossQPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: str | float = "auto",
        target_entropy: str | float = "auto",
        policy_delay: int = 3,
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
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            1.0,  # no target networks, tau=1.0
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
        self.log_ent_coef = None  # type: Optional[torch.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.ent_coef_optimizer: torch.optim.Adam | None = None
        self.policy_delay = policy_delay

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore[union-attr]
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

    # ------------------------------------------------------------------
    # Lightning hooks (manual optimization)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        return optimizers

    def training_step(self, batch: Any, batch_idx: int) -> None:
        """Off-policy CrossQ step: collect env transitions then do gradient updates."""
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
        """CrossQ gradient updates (manual optimization with BatchNorm-aware forward pass)."""
        optimizers = self.optimizers()
        actor_opt = optimizers[0]  # ty:ignore[not-subscriptable]
        critic_opt = optimizers[1]  # ty:ignore[not-subscriptable]
        ent_opt = optimizers[2] if len(optimizers) > 2 else None  # ty:ignore[invalid-argument-type, not-subscriptable]

        self.policy.set_training_mode(True)
        base_optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            base_optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(base_optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            ent_coef = torch.exp(self.log_ent_coef.detach()) if self.log_ent_coef is not None else self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            with torch.no_grad():
                self.actor.set_bn_training_mode(False)
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

            # Joint forward pass for BatchNorm statistics
            all_obs = torch.cat([replay_data.observations, replay_data.next_observations], dim=0)
            all_actions = torch.cat([replay_data.actions, next_actions], dim=0)
            self.critic.set_bn_training_mode(True)
            all_q_values = torch.cat(self.critic(all_obs, all_actions), dim=1)
            self.critic.set_bn_training_mode(False)
            current_q_values, next_q_values = torch.split(all_q_values, batch_size, dim=0)
            current_q_values = current_q_values.T[..., None]

            with torch.no_grad():
                next_q_values, _ = torch.min(next_q_values.detach(), dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values.detach()) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())
            critic_opt.zero_grad()
            self.manual_backward(critic_loss)
            critic_opt.step()

            # Delayed updates (actor + entropy coef)
            if self._n_updates % self.policy_delay == 0:
                self.actor.set_bn_training_mode(True)
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
                self.actor.set_bn_training_mode(False)

                if ent_opt is not None and self.log_ent_coef is not None:
                    assert isinstance(self.target_entropy, float)
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                    ent_opt.zero_grad()
                    self.manual_backward(ent_coef_loss)
                    ent_opt.step()

                self.critic.set_bn_training_mode(False)
                q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())
                actor_opt.zero_grad()
                self.manual_backward(actor_loss)
                actor_opt.step()

        self.log("train/n_updates", float(self._n_updates), on_step=True, on_epoch=False)
        self.log("train/ent_coef", float(np.mean(ent_coefs)), on_step=True, on_epoch=False)
        self.log("train/critic_loss", float(np.mean(critic_losses)), on_step=True, on_epoch=False)
        if actor_losses:
            self.log("train/actor_loss", float(np.mean(actor_losses)), on_step=True, on_epoch=False)
        if ent_coef_losses:
            self.log("train/ent_coef_loss", float(np.mean(ent_coef_losses)), on_step=True, on_epoch=False)

    def learn(
        self: SelfCrossQ,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 4,
        tb_log_name: str = "CrossQ",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfCrossQ:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "actor", "critic"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
