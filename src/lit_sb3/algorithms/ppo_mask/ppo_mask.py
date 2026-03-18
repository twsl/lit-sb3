from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, Union

from gymnasium import spaces
import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from lit_sb3.algorithms.on_policy import OnPolicyAlgorithm
from lit_sb3.algorithms.ppo_mask.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from lit_sb3.buffers import RolloutBuffer
from lit_sb3.buffers.maskable.dict_rollout import MaskableDictRolloutBuffer
from lit_sb3.buffers.maskable.rollout import MaskableRolloutBuffer
from lit_sb3.callbacks import BaseCallback
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.core.utils import FloatSchedule, explained_variance, obs_as_tensor
from lit_sb3.policies import BasePolicy
from lit_sb3.policies.maskable.actor_critic import MaskableActorCriticPolicy
from lit_sb3.utils.maskable import get_action_masks, is_masking_supported
from lit_sb3.vec_env import VecEnv

SelfMaskablePPO = TypeVar("SelfMaskablePPO", bound="MaskablePPO")


class MaskablePPO(OnPolicyAlgorithm):
    """Proximal Policy Optimization algorithm (PPO) (clip version) with Invalid Action Masking.

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    Background on Invalid Action Masking: https://arxiv.org/abs/2006.14171

    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from (if registered in Gym, can be str)
        learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        n_steps: The number of steps to run for each environment per update
            (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
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
        target_kl: Limit the KL divergence between updates,
            because the clipping is not enough to prevent large update
            see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
            By default, there is no limit on the kl div.
        stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        tensorboard_log: the log location for tensorboard (if None, no logging)
        policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_mask_policies`
        verbose: the verbosity level: 0 no output, 1 info, 2 debug
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
    policy: MaskableActorCriticPolicy
    rollout_buffer: MaskableRolloutBuffer

    def __init__(
        self,
        policy: str | type[MaskableActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int | None = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
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
    ):
        super().__init__(
            policy,  # type: ignore[arg-type]
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        # Lightning DataLoader config
        self._rl_batch_size = batch_size
        self._rl_n_epochs = n_epochs

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, MaskableActorCriticPolicy):
            raise TypeError("Policy must subclass MaskableActorCriticPolicy")

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = MaskableDictRolloutBuffer
            else:
                self.rollout_buffer_class = MaskableRolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(  # type: ignore[assignment]
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """Collect experiences using the current policy and fill a ``RolloutBuffer``.

        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        This method is largely identical to the implementation found in the parent class.

        Args:
            env: The training environment
            callback: Callback that will be called at each step
                (and at the beginning and end of the rollout)
            rollout_buffer: Buffer to fill with rollouts
            n_steps: Number of experiences to collect per environment
            use_masking: Whether or not to use invalid action masks during training

            n_rollout_steps: Number of steps to rollout.

        Returns:
            True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)), (
            "RolloutBuffer doesn't support action masking"
        )
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]

                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
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
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Get the policy action from an observation (and optional hidden state).

        Includes sugar-coating to handle different observations (e.g. normalizing images).

        Args:
            observation: the input observation
            state: The last hidden states (can be None, used in recurrent policies)
            episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            deterministic: Whether or not to return deterministic actions.

            action_masks: Boolean masks for invalid actions (True = valid).

        Returns:
            the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, deterministic, action_masks=action_masks)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Collect rollouts with masking, then set policy to train mode."""
        super().on_train_epoch_start()  # collect_rollouts uses use_masking=True by default
        self.policy.set_training_mode(True)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute MaskablePPO clipped-surrogate loss for one mini-batch."""
        rollout_data = batch

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = rollout_data.actions.long().flatten()

        values, log_prob, entropy = self.policy.evaluate_actions(
            rollout_data.observations,
            actions,
            action_masks=rollout_data.action_masks,
        )
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

        # Gradient clipping (Lightning runs backward automatically)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

        # KL divergence early stopping
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
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Log epoch-level stats."""
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.log("train/explained_variance", float(explained_var), on_step=False, on_epoch=True)
        self.log("train/n_updates", float(self._n_updates), on_step=False, on_epoch=True)

    def learn(  # type: ignore[override]
        self: SelfMaskablePPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 1,
        tb_log_name: str = "MaskablePPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfMaskablePPO:
        # Note: use_masking is stored so on_train_epoch_start can pass it to collect_rollouts.
        # Default is True; MaskablePPO.collect_rollouts also defaults to use_masking=True.
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
