from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union

from gymnasium import spaces
import torch
from torch.nn import functional as F  # noqa: N812

from lit_sb3.algorithms.on_policy import OnPolicyAlgorithm
from lit_sb3.buffers import RolloutBuffer
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.core.utils import explained_variance
from lit_sb3.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy

SelfA2C = TypeVar("SelfA2C", bound="A2C")


class A2C(OnPolicyAlgorithm):
    """Advantage Actor Critic (A2C).

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from (if registered in Gym, can be str)
        learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        n_steps: The number of steps to run for each environment per update
            (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
            Equivalent to classic advantage when set to 1.
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        max_grad_norm: The maximum value for the gradient clipping
        rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
            of RMSProp update
        use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
        use_sde: Whether to use generalized State Dependent Exploration (gSDE)
            instead of action noise exploration (default: False)
        sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
        rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
        normalize_advantage: Whether to normalize or not the advantage
        stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        tensorboard_log: the log location for tensorboard (if None, no logging)
        policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`a2c_policies`
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
        learning_rate: float | Schedule = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        normalize_advantage: bool = False,
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

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = torch.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = {"alpha": 0.99, "eps": rms_prop_eps, "weight_decay": 0}

        if _init_setup_model:
            self._setup_model()

        # A2C uses the full rollout buffer in a single batch
        self._rl_batch_size = None
        self._rl_n_epochs = 1

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return self.policy.optimizer

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.policy.set_training_mode(True)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """A2C loss on the full rollout buffer (single gradient step)."""
        rollout_data = batch

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages * log_prob).mean()
        value_loss = F.mse_loss(rollout_data.returns, values)
        entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        self._n_updates += 1
        self.log_dict(
            {
                "train/policy_loss": policy_loss.item(),
                "train/value_loss": value_loss.item(),
                "train/entropy_loss": entropy_loss.item(),
                "train/loss": loss.item(),
                "train/n_updates": float(self._n_updates),
            },
            on_step=True,
            on_epoch=False,
        )
        if hasattr(self.policy, "log_std"):
            self.log("train/std", torch.exp(self.policy.log_std).mean().item(), on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.log("train/explained_variance", float(explained_var), on_step=False, on_epoch=True)

    def learn(
        self: SelfA2C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 100,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2C:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
