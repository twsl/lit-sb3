import copy
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union
import warnings

from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from lit_sb3.algorithms.on_policy import OnPolicyAlgorithm
from lit_sb3.algorithms.trpo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from lit_sb3.buffers import RolloutBuffer
from lit_sb3.core.types import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from lit_sb3.core.utils import conjugate_gradient_solver, explained_variance, flat_grad
from lit_sb3.distributions import kl_divergence
from lit_sb3.policies import ActorCriticPolicy, BasePolicy

SelfTRPO = TypeVar("SelfTRPO", bound="TRPO")


class TRPO(OnPolicyAlgorithm):
    """Trust Region Policy Optimization (TRPO).

    Paper: https://arxiv.org/abs/1502.05477
    Adapted from sb3-contrib to use PyTorch Lightning.
    """

    automatic_optimization = False  # manual optimization for TRPO (custom CG update)

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        sub_sampling_factor: int = 1,
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
            ent_coef=0.0,
            vf_coef=0.0,
            max_grad_norm=0.0,
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
        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            if normalize_advantage:
                assert buffer_size > 1, (
                    "`n_steps * n_envs` must be greater than 1. "
                    f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
                )
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size}, "
                    f"but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`, "
                    f"after every {untruncated_batches} untruncated mini-batches, "
                    f"there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})",
                    stacklevel=2,
                )

        self.batch_size = batch_size
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        self.line_search_shrinking_factor = line_search_shrinking_factor
        self.line_search_max_iter = line_search_max_iter
        self.target_kl = target_kl
        self.n_critic_updates = n_critic_updates
        self.sub_sampling_factor = sub_sampling_factor
        # TRPO processes full buffer in one pass
        self._rl_batch_size = None
        self._rl_n_epochs = 1

        if _init_setup_model:
            self._setup_model()

    def _compute_actor_grad(
        self, kl_div: torch.Tensor, policy_objective: torch.Tensor
    ) -> tuple[list[nn.Parameter], torch.Tensor, torch.Tensor, list[tuple[int, ...]]]:
        policy_objective_gradients_list = []
        grad_kl_list = []
        grad_shape: list[tuple[int, ...]] = []
        actor_params: list[nn.Parameter] = []

        for name, param in self.policy.named_parameters():
            if "value" in name:
                continue
            kl_param_grad, *_ = torch.autograd.grad(
                kl_div,
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            if kl_param_grad is not None:
                policy_objective_grad, *_ = torch.autograd.grad(
                    policy_objective, param, retain_graph=True, only_inputs=True
                )
                grad_shape.append(kl_param_grad.shape)
                grad_kl_list.append(kl_param_grad.reshape(-1))
                policy_objective_gradients_list.append(policy_objective_grad.reshape(-1))
                actor_params.append(param)

        policy_objective_gradients = torch.cat(policy_objective_gradients_list)
        grad_kl = torch.cat(grad_kl_list)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape

    # ------------------------------------------------------------------
    # Lightning hooks (manual optimization — actor uses CG, not autograd backward)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return self.policy.optimizer

    def on_train_epoch_start(self) -> None:
        """Configure TRPO to process full buffer in one pass, then collect rollout."""
        super().on_train_epoch_start()
        self.policy.set_training_mode(True)

    def training_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore[override]
        """TRPO update: CG actor update + critic gradient descent."""
        opt = self.optimizers()
        rollout_data = batch

        policy_objective_values: list[float] = []
        kl_divergences: list[float] = []
        line_search_results: list[bool] = []
        value_losses: list[float] = []

        # --- Actor update (trust region via conjugate gradient + line search) ---
        # Optional sub-sampling for faster computation
        if self.sub_sampling_factor > 1:
            rollout_data = RolloutBufferSamples(
                rollout_data.observations[:: self.sub_sampling_factor],
                rollout_data.actions[:: self.sub_sampling_factor],
                None,  # type: ignore[arg-type]
                rollout_data.old_log_prob[:: self.sub_sampling_factor],
                rollout_data.advantages[:: self.sub_sampling_factor],
                None,  # type: ignore[arg-type]
            )

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = rollout_data.actions.long().flatten()

        with torch.no_grad():
            old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

        distribution = self.policy.get_distribution(rollout_data.observations)
        log_prob = distribution.log_prob(actions)

        advantages = rollout_data.advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - rollout_data.old_log_prob)
        policy_objective = (advantages * ratio).mean()
        kl_div = kl_divergence(distribution, old_distribution).mean()

        opt.zero_grad()  # ty:ignore[unresolved-attribute]
        actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
            kl_div, policy_objective
        )

        hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)
        search_direction = conjugate_gradient_solver(
            hessian_vector_product_fn,
            policy_objective_gradients,
            max_iter=self.cg_max_steps,
        )

        line_search_max_step_size = 2 * self.target_kl
        line_search_max_step_size /= torch.matmul(
            search_direction, hessian_vector_product_fn(search_direction, retain_graph=False)
        )
        line_search_max_step_size = torch.sqrt(line_search_max_step_size)

        line_search_backtrack_coeff = 1.0
        original_actor_params = [param.detach().clone() for param in actor_params]
        is_line_search_success = False

        with torch.no_grad():
            for _ in range(self.line_search_max_iter):
                start_idx = 0
                for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape, strict=False):
                    n_params = param.numel()
                    param.data = (
                        original_param.data
                        + line_search_backtrack_coeff
                        * line_search_max_step_size
                        * search_direction[start_idx : (start_idx + n_params)].view(shape)
                    )
                    start_idx += n_params

                distribution = self.policy.get_distribution(rollout_data.observations)
                log_prob = distribution.log_prob(actions)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                new_policy_objective = (advantages * ratio).mean()
                kl_div = kl_divergence(distribution, old_distribution).mean()

                if (kl_div < self.target_kl) and (new_policy_objective > policy_objective):
                    is_line_search_success = True
                    break

                line_search_backtrack_coeff *= self.line_search_shrinking_factor

            line_search_results.append(is_line_search_success)

            if not is_line_search_success:
                for param, original_param in zip(actor_params, original_actor_params, strict=False):
                    param.data = original_param.data.clone()
                policy_objective_values.append(policy_objective.item())
                kl_divergences.append(0.0)
            else:
                policy_objective_values.append(new_policy_objective.item())
                kl_divergences.append(kl_div.item())

        # --- Critic update (standard gradient descent) ---
        for _ in range(self.n_critic_updates):
            for critic_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(critic_data.observations)
                value_loss = F.mse_loss(critic_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                opt.zero_grad()  # ty:ignore[unresolved-attribute]
                self.manual_backward(value_loss)
                # Zero out actor param gradients (shared param protection)
                for param in actor_params:
                    param.grad = None
                opt.step()  # ty:ignore[unresolved-attribute]

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.log_dict(
            {
                "train/policy_objective": float(np.mean(policy_objective_values)),
                "train/value_loss": float(np.mean(value_losses)),
                "train/kl_divergence_loss": float(np.mean(kl_divergences)),
                "train/explained_variance": float(explained_var),
                "train/is_line_search_success": float(np.mean(line_search_results)),
                "train/n_updates": float(self._n_updates),
            },
            on_step=True,
            on_epoch=False,
        )
        if hasattr(self.policy, "log_std"):
            self.log("train/std", float(torch.exp(self.policy.log_std).mean().item()), on_step=True)

    def hessian_vector_product(
        self,
        params: list[nn.Parameter],
        grad_kl: torch.Tensor,
        vector: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Compute the matrix-vector product with the Fisher information matrix."""
        jacobian_vector_product = torch.dot(grad_kl, vector)
        return flat_grad(jacobian_vector_product, params, retain_graph=retain_graph) + self.cg_damping * vector

    def learn(
        self: SelfTRPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 1,
        tb_log_name: str = "TRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTRPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
