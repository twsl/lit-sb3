import copy
from functools import partial
from pathlib import Path
import sys
import time
from typing import Any, ClassVar, Optional, TypeVar, Union
import warnings

from gymnasium import spaces
import numpy as np
import torch
import torch.nn.utils

from lit_sb3.algorithms.ars.policies import ARSPolicy, LinearPolicy, MlpPolicy
from lit_sb3.algorithms.base import BaseAlgorithm
from lit_sb3.callbacks import BaseCallback
from lit_sb3.core.evaluation import evaluate_policy
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule
from lit_sb3.policies import BasePolicy
from lit_sb3.utils.math import safe_mean
from lit_sb3.utils.save import load_from_zip_file
from lit_sb3.utils.schedule import FloatSchedule
from lit_sb3.vec_env.async_eval import AsyncEval

SelfARS = TypeVar("SelfARS", bound="ARS")


class ARS(BaseAlgorithm):
    """Augmented Random Search: https://arxiv.org/abs/1803.07055.

    Original implementation: https://github.com/modestyachts/ARS
    C++/Cuda Implementation: https://github.com/google-research/tiny-differentiable-simulator/
    150 LOC Numpy Implementation: https://github.com/alexis-jacq/numpy_ARS/blob/master/asr.py

    Args:
        policy: The policy to train, can be an instance of ``ARSPolicy``, or a string from ["LinearPolicy", "MlpPolicy"]
        env: The environment to train on, may be a string if registered with gym
        n_delta: How many random perturbations of the policy to try at each update step.
        n_top: How many of the top delta to use in each update step. Default is n_delta
        learning_rate: Float or schedule for the step size
        delta_std: Float or schedule for the exploration noise
        zero_policy: Boolean determining if the passed policy should have it's weights zeroed before training.
        alive_bonus_offset: Constant added to the reward at each step, used to cancel out alive bonuses.
        n_eval_episodes: Number of episodes to evaluate each candidate.
        policy_kwargs: Keyword arguments to pass to the policy on creation. See :ref:`ars_policies`
        stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
            the reported success rate, mean episode length, and mean reward over
        tensorboard_log: String with the directory to put tensorboard logs:
        seed: Random seed for the training
        verbose: Verbosity level: 0 no output, 1 info, 2 debug
        device: Torch device to use for training, defaults to "cpu"
        _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "LinearPolicy": LinearPolicy,
    }

    def __init__(
        self,
        policy: str | type[ARSPolicy],
        env: GymEnv | str,
        n_delta: int = 8,
        n_top: int | None = None,
        learning_rate: float | Schedule = 0.02,
        delta_std: float | Schedule = 0.05,
        zero_policy: bool = True,
        alive_bonus_offset: float = 0,
        n_eval_episodes: int = 1,
        policy_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | Path | None = None,
        seed: int | None = None,
        verbose: int = 0,
        device: torch.device | str = "cpu",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            supported_action_spaces=(spaces.Box, spaces.Discrete),
            support_multi_env=True,
            seed=seed,
        )

        self.n_delta = n_delta
        self.pop_size = 2 * n_delta
        self.delta_std_schedule = FloatSchedule(delta_std)
        self.n_eval_episodes = n_eval_episodes

        if n_top is None:
            n_top = n_delta

        # Make sure our hyper parameters are valid and auto correct them if they are not
        if n_top > n_delta:
            warnings.warn(f"n_top = {n_top} > n_delta = {n_top}, setting n_top = n_delta", stacklevel=2)
            n_top = n_delta

        self.n_top = n_top

        self.alive_bonus_offset = alive_bonus_offset
        self.zero_policy = zero_policy
        self.weights = None  # Need to call init model to initialize weight
        self.processes = None
        # Keep track of how many steps where elapsed before a new rollout
        # Important for syncing observation normalization between workers
        self.old_count = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(self.observation_space, self.action_space, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self.weights = torch.nn.utils.parameters_to_vector(self.policy.parameters()).detach()
        self.n_params = len(self.weights)

        if self.zero_policy:
            self.weights = torch.zeros_like(self.weights, requires_grad=False)
            self.policy.load_from_vector(self.weights.cpu())  # ty:ignore[invalid-argument-type]

    def _mimic_monitor_wrapper(self, episode_rewards: np.ndarray, episode_lengths: np.ndarray) -> None:
        """Helper to mimic Monitor wrapper and report episode statistics (mean reward, mean episode length).

        Args:
            episode_rewards: List containing per-episode rewards
            episode_lengths: List containing per-episode lengths (in number of steps)
        """
        # Mimic Monitor Wrapper
        infos = [
            {"episode": {"r": episode_reward, "l": episode_length}}
            for episode_reward, episode_length in zip(episode_rewards, episode_lengths, strict=False)
        ]

        self._update_info_buffer(infos)

    def _trigger_callback(
        self,
        _locals: dict[str, Any],
        _globals: dict[str, Any],
        callback: BaseCallback,
        n_envs: int,
    ) -> None:
        """Callback passed to the ``evaluate_policy()`` helper.

        in order to increment the number of timesteps
        and trigger events in the single process version.

        Args:
            _locals:
            _globals:
            callback: Callback that will be called at every step
            n_envs: Number of environments
        """
        self.num_timesteps += n_envs
        callback.on_step()

    def evaluate_candidates(
        self, candidate_weights: torch.Tensor, callback: BaseCallback, async_eval: AsyncEval | None
    ) -> torch.Tensor:
        """Evaluate each candidate.

        Args:
            candidate_weights: The candidate weights to be evaluated.
            callback: Callback that will be called at each step
                (or after evaluation in the multiprocess version)
            async_eval: The object for asynchronous evaluation of candidates.

        Returns:
            The episodic return for each candidate.
        """
        batch_steps = 0
        # returns == sum of rewards
        candidate_returns = torch.zeros(self.pop_size, device=self.device)
        train_policy = copy.deepcopy(self.policy)
        # Empty buffer to show only mean over one iteration (one set of candidates) in the logs
        self.ep_info_buffer = []
        callback.on_rollout_start()

        if async_eval is not None:
            # Multiprocess asynchronous version
            async_eval.send_jobs(candidate_weights, self.pop_size)
            results = async_eval.get_results()

            for weights_idx, (episode_rewards, episode_lengths) in results:
                # Update reward to cancel out alive bonus if needed
                candidate_returns[weights_idx] = sum(episode_rewards) + self.alive_bonus_offset * sum(episode_lengths)
                batch_steps += np.sum(episode_lengths)
                self._mimic_monitor_wrapper(episode_rewards, episode_lengths)

            # Combine the filter stats of each process for normalization
            for worker_obs_rms in async_eval.get_obs_rms():
                if self._vec_normalize_env is not None:
                    # worker_obs_rms.count -= self.old_count
                    self._vec_normalize_env.obs_rms.combine(worker_obs_rms)  # ty:ignore[unresolved-attribute]
                    # Hack: don't count timesteps twice (between the two are synced)
                    # otherwise it will lead to overflow,
                    # in practice we would need two RunningMeanStats
                    self._vec_normalize_env.obs_rms.count -= self.old_count  # ty:ignore[unresolved-attribute]

            # Synchronise VecNormalize if needed
            if self._vec_normalize_env is not None:
                async_eval.sync_obs_rms(self._vec_normalize_env.obs_rms.copy())  # ty:ignore[invalid-argument-type]
                self.old_count = self._vec_normalize_env.obs_rms.count  # ty:ignore[unresolved-attribute]

            # Hack to have Callback events
            for _ in range(batch_steps // len(async_eval.remotes)):
                self.num_timesteps += len(async_eval.remotes)
                callback.on_step()
        else:
            # Single process, synchronous version
            for weights_idx in range(self.pop_size):
                # Load current candidate weights
                train_policy.load_from_vector(candidate_weights[weights_idx].cpu())  # ty:ignore[invalid-argument-type]
                # Evaluate the candidate
                episode_rewards, episode_lengths = evaluate_policy(
                    train_policy,
                    self.env,  # ty:ignore[invalid-argument-type]
                    n_eval_episodes=self.n_eval_episodes,
                    return_episode_rewards=True,
                    # Increment num_timesteps too (slight mismatch with multi envs)
                    callback=partial(self._trigger_callback, callback=callback, n_envs=self.env.num_envs),  # ty:ignore[unresolved-attribute]
                    warn=False,
                )
                # Update reward to cancel out alive bonus if needed
                candidate_returns[weights_idx] = sum(episode_rewards) + self.alive_bonus_offset * sum(episode_lengths)  # ty:ignore[no-matching-overload]
                batch_steps += sum(episode_lengths)  # ty:ignore[no-matching-overload]
                self._mimic_monitor_wrapper(episode_rewards, episode_lengths)  # ty:ignore[invalid-argument-type]

            # Note: we increment the num_timesteps inside the evaluate_policy()
            # however when using multiple environments, there will be a slight
            # mismatch between the number of timesteps used and the number
            # of calls to the step() method (cf. implementation of evaluate_policy())
            # self.num_timesteps += batch_steps

        callback.on_rollout_end()

        return candidate_returns

    def dump_logs(self) -> None:
        """Dump information to the logger."""
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:  # ty:ignore[invalid-argument-type, not-subscriptable]
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))  # ty:ignore[not-iterable]
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))  # ty:ignore[not-iterable]
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

    def _do_one_update(self, callback: BaseCallback, async_eval: AsyncEval | None) -> None:
        """Sample new candidates, evaluate them and then update current policy.

        Args:
            callback: callback(s) called at every step with state of the algorithm.
            async_eval: The object for asynchronous evaluation of candidates.
        """
        # Retrieve current parameter noise standard deviation
        # and current learning rate
        delta_std = self.delta_std_schedule(self._current_progress_remaining)
        learning_rate = self.lr_schedule(self._current_progress_remaining)
        # Sample the parameter noise, it will be scaled by delta_std
        deltas = torch.normal(mean=0.0, std=1.0, size=(self.n_delta, self.n_params), device=self.device)
        policy_deltas = deltas * delta_std
        # Generate 2 * n_delta candidate policies by adding noise to the current weights
        candidate_weights = torch.cat([self.weights + policy_deltas, self.weights - policy_deltas])  # ty:ignore[unsupported-operator]

        with torch.no_grad():
            candidate_returns = self.evaluate_candidates(candidate_weights, callback, async_eval)

        # Returns corresponding to weights + deltas
        plus_returns = candidate_returns[: self.n_delta]
        # Returns corresponding to weights - deltas
        minus_returns = candidate_returns[self.n_delta :]

        # Keep only the top performing candidates for update
        top_returns, _ = torch.max(torch.vstack((plus_returns, minus_returns)), dim=0)
        top_idx = torch.argsort(top_returns, descending=True)[: self.n_top]
        plus_returns = plus_returns[top_idx]
        minus_returns = minus_returns[top_idx]
        deltas = deltas[top_idx]

        # Scale learning rate by the return standard deviation:
        # take smaller steps when there is a high variance in the returns
        return_std = torch.cat([plus_returns, minus_returns]).std()
        step_size = learning_rate / (self.n_top * return_std + 1e-6)
        # Approximate gradient step
        self.weights = self.weights + step_size * ((plus_returns - minus_returns) @ deltas)  # ty:ignore[unsupported-operator]
        self.policy.load_from_vector(self.weights.cpu())

        self.logger.record("train/iterations", self._n_updates, exclude="tensorboard")
        self.logger.record("train/delta_std", delta_std)
        self.logger.record("train/learning_rate", learning_rate)
        self.logger.record("train/step_size", step_size.item())
        self.logger.record("rollout/return_std", return_std.item())

        self._n_updates += 1

    def learn(
        self: SelfARS,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 1,
        tb_log_name: str = "ARS",
        reset_num_timesteps: bool = True,
        async_eval: AsyncEval | None = None,
        progress_bar: bool = False,
    ) -> SelfARS:  # ty:ignore[invalid-method-override]
        """Return a trained model.

        Args:
            total_timesteps: The total number of samples (env steps) to train on
            callback: callback(s) called at every step with state of the algorithm.
            log_every_n_steps: The number of timesteps before logging.
            tb_log_name: the name of the run for TensorBoard logging
            reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
            async_eval: The object for asynchronous evaluation of candidates.
            progress_bar: Display a progress bar using tqdm and rich.

        Returns:
            the trained model
        """
        total_steps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_steps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            self._do_one_update(callback, async_eval)
            if log_every_n_steps is not None and self._n_updates % log_every_n_steps == 0:
                self.dump_logs()

        if async_eval is not None:
            async_eval.close()

        callback.on_training_end()

        return self

    def set_parameters(
        self,
        load_path_or_dict: str | dict[str, dict],
        exact_match: bool = True,
        device: torch.device | str = "auto",
    ) -> None:  # ty:ignore[invalid-method-override]
        # Patched set_parameters() to handle ARS linear policy saved with sb3-contrib < 1.7.0
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Patch to load LinearPolicy saved using sb3-contrib < 1.7.0
        # See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/122#issuecomment-1331981230
        for name in {"weight", "bias"}:
            if f"action_net.{name}" in params.get("policy", {}):
                params["policy"][f"action_net.0.{name}"] = params["policy"][f"action_net.{name}"]  # ty:ignore[invalid-argument-type, invalid-assignment]
                del params["policy"][f"action_net.{name}"]  # ty:ignore[not-subscriptable]

        super().set_parameters(params, exact_match=exact_match)  # ty:ignore[invalid-argument-type]
