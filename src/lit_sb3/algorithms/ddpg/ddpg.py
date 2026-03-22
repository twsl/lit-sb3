from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import torch

from lit_sb3.algorithms.td3.policies import TD3Policy
from lit_sb3.algorithms.td3.td3 import TD3
from lit_sb3.buffers import ReplayBuffer
from lit_sb3.core.noise import ActionNoise
from lit_sb3.core.types import GymEnv, MaybeCallback, Schedule

SelfDDPG = TypeVar("SelfDDPG", bound="DDPG")


class DDPG(TD3):
    """Deep Deterministic Policy Gradient (DDPG).

    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Note: we treat DDPG as a special case of its successor TD3.

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
        policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ddpg_policies`
        verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        seed: Seed for the pseudo random generators
        device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: str | type[TD3Policy],
        learning_rate: float | Schedule = 1e-3,
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
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: torch.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model and hasattr(self, "observation_space"):
            self._setup_model()

    def learn(
        self: SelfDDPG,
        total_timesteps: int,
        env: GymEnv | str | None = None,
        callback: MaybeCallback = None,
        log_every_n_steps: int | None = 4,
        tb_log_name: str = "DDPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDPG:
        return super().learn(
            total_timesteps=total_timesteps,
            env=env,
            callback=callback,
            log_every_n_steps=log_every_n_steps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
