"""Optuna hyperparameter samplers for all supported algorithms.

Adapted from
https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py

Requires ``optuna`` to be installed::

    pip install optuna
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import nn as nn

from lit_sb3.core.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

try:
    import optuna

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Converter helpers (sampled → actual hyperparams)
# ---------------------------------------------------------------------------


def convert_onpolicy_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    """Convert sampled on-policy parameter dict to algorithm kwargs."""
    hyperparams = sampled_params.copy()

    hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
    del hyperparams["one_minus_gamma"]

    hyperparams["gae_lambda"] = 1 - sampled_params["one_minus_gae_lambda"]
    del hyperparams["one_minus_gae_lambda"]

    net_arch_key = sampled_params["net_arch"]
    del hyperparams["net_arch"]

    for name in ["batch_size", "n_steps"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    net_arch = {
        "tiny": {"pi": [64], "vf": [64]},
        "small": {"pi": [64, 64], "vf": [64, 64]},
        "medium": {"pi": [256, 256], "vf": [256, 256]},
    }[net_arch_key]

    activation_fn_name = sampled_params["activation_fn"]
    del hyperparams["activation_fn"]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    return {
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
        **hyperparams,
    }


def convert_offpolicy_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    """Convert sampled off-policy parameter dict to algorithm kwargs."""
    hyperparams = sampled_params.copy()

    hyperparams["gamma"] = 1 - sampled_params["one_minus_gamma"]
    del hyperparams["one_minus_gamma"]

    net_arch_key = sampled_params["net_arch"]
    del hyperparams["net_arch"]

    for name in ["batch_size"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "large": [256, 256, 256],
        "verybig": [512, 512, 512],
    }[net_arch_key]

    if "train_freq" in sampled_params:
        hyperparams["gradient_steps"] = sampled_params["train_freq"]
        if "subsample_steps" in sampled_params:
            hyperparams["gradient_steps"] = max(sampled_params["train_freq"] // sampled_params["subsample_steps"], 1)
            del hyperparams["subsample_steps"]

    hyperparams["policy_kwargs"] = hyperparams.get("policy_kwargs", {})
    hyperparams["policy_kwargs"]["net_arch"] = net_arch

    if "activation_fn" in sampled_params:
        activation_fn_name = sampled_params["activation_fn"]
        del hyperparams["activation_fn"]
        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }[activation_fn_name]
        hyperparams["policy_kwargs"]["activation_fn"] = activation_fn

    # TQC / QRDQN
    if "n_quantiles" in sampled_params:
        del hyperparams["n_quantiles"]
        hyperparams["policy_kwargs"].update({"n_quantiles": sampled_params["n_quantiles"]})

    return hyperparams


def convert_ars_params(sampled_params: dict[str, Any]) -> dict[str, Any]:
    """Convert sampled ARS parameter dict to algorithm kwargs."""
    hyperparams = sampled_params.copy()

    for name in ["n_delta"]:
        if f"{name}_pow" in sampled_params:
            hyperparams[name] = 2 ** sampled_params[f"{name}_pow"]
            del hyperparams[f"{name}_pow"]

    top_frac_size = sampled_params["top_frac_size"]
    hyperparams["n_top"] = max(int(top_frac_size * hyperparams["n_delta"]), 1)
    del hyperparams["top_frac_size"]

    return hyperparams


# ---------------------------------------------------------------------------
# Per-algorithm samplers
# ---------------------------------------------------------------------------


def sample_a2c_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample A2C hyperparameters."""
    n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256, 512, 1024, 2048])
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)

    return convert_onpolicy_params(
        {
            "n_steps": n_steps,
            "one_minus_gamma": one_minus_gamma,
            "one_minus_gae_lambda": one_minus_gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }
    )


def sample_ppo_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample PPO hyperparameters."""
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("n_steps", 2**n_steps_pow)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    return convert_onpolicy_params(
        {
            "n_steps_pow": n_steps_pow,
            "batch_size_pow": batch_size_pow,
            "one_minus_gamma": one_minus_gamma,
            "one_minus_gae_lambda": one_minus_gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "n_epochs": n_epochs,
            "max_grad_norm": max_grad_norm,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }
    )


def sample_ppo_lstm_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample RecurrentPPO (PPO-LSTM) hyperparameters."""
    return sample_ppo_params(trial, n_actions, n_envs, additional_args)


def sample_trpo_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample TRPO hyperparameters."""
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    n_critic_updates = trial.suggest_int("n_critic_updates", 5, 30)
    cg_max_steps = trial.suggest_int("cg_max_steps", 5, 30)
    target_kl = trial.suggest_float("target_kl", 0.001, 0.1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("n_steps", 2**n_steps_pow)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    return convert_onpolicy_params(
        {
            "n_steps_pow": n_steps_pow,
            "batch_size_pow": batch_size_pow,
            "one_minus_gamma": one_minus_gamma,
            "one_minus_gae_lambda": one_minus_gae_lambda,
            "learning_rate": learning_rate,
            "cg_max_steps": cg_max_steps,
            "target_kl": target_kl,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "n_critic_updates": n_critic_updates,
        }
    )


def _sample_her_params(
    trial: Any,
    hyperparams: dict[str, Any],
    her_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Add HER-specific hyperparameters to *hyperparams*."""
    her_kwargs = her_kwargs.copy()
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams


def sample_sac_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample SAC hyperparameters."""
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 11)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    tau = trial.suggest_float("tau", 0.001, 0.08, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    hyperparams: dict[str, Any] = {
        "one_minus_gamma": one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size_pow": batch_size_pow,
        "train_freq": train_freq,
        "tau": tau,
        "net_arch": net_arch,
    }

    if additional_args.get("using_her_replay_buffer"):
        hyperparams = _sample_her_params(trial, hyperparams, additional_args.get("her_kwargs", {}))

    if "sample_tqc" in additional_args:
        n_quantiles = trial.suggest_int("n_quantiles", 5, 50)
        top_quantiles_to_drop_per_net = trial.suggest_int("top_quantiles_to_drop_per_net", 0, min(n_quantiles - 1, 5))
        hyperparams.update(
            {
                "n_quantiles": n_quantiles,
                "top_quantiles_to_drop_per_net": top_quantiles_to_drop_per_net,
            }
        )

    return convert_offpolicy_params(hyperparams)


def sample_tqc_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample TQC hyperparameters (TQC = SAC + distributional RL)."""
    additional_args["sample_tqc"] = True
    return sample_sac_params(trial, n_actions, n_envs, additional_args)


def sample_td3_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample TD3 / DDPG hyperparameters."""
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 11)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    tau = trial.suggest_float("tau", 0.001, 0.08, log=True)
    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_float("noise_std", 0, 0.5)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    hyperparams: dict[str, Any] = {
        "one_minus_gamma": one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size_pow": batch_size_pow,
        "train_freq": train_freq,
        "tau": tau,
        "net_arch": net_arch,
    }

    if noise_type == "normal":
        assert NormalActionNoise is not None
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions),
        )
    elif noise_type == "ornstein-uhlenbeck":
        assert OrnsteinUhlenbeckActionNoise is not None
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_std * np.ones(n_actions),
        )

    if additional_args.get("using_her_replay_buffer"):
        hyperparams = _sample_her_params(trial, hyperparams, additional_args.get("her_kwargs", {}))

    return convert_offpolicy_params(hyperparams)


def sample_dqn_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample DQN hyperparameters."""
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    batch_size_pow = trial.suggest_int("batch_size_pow", 2, 11)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    subsample_steps = trial.suggest_int("subsample_steps", 1, min(train_freq, 8))
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_int("target_update_interval", 1, 20_000, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    trial.set_user_attr("gamma", 1 - one_minus_gamma)
    trial.set_user_attr("batch_size", 2**batch_size_pow)

    hyperparams: dict[str, Any] = {
        "one_minus_gamma": one_minus_gamma,
        "learning_rate": learning_rate,
        "batch_size_pow": batch_size_pow,
        "train_freq": train_freq,
        "subsample_steps": subsample_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "net_arch": net_arch,
    }

    if additional_args.get("using_her_replay_buffer"):
        hyperparams = _sample_her_params(trial, hyperparams, additional_args.get("her_kwargs", {}))

    if "sample_qrdqn" in additional_args:
        hyperparams["n_quantiles"] = trial.suggest_int("n_quantiles", 5, 200)

    return convert_offpolicy_params(hyperparams)


def sample_qrdqn_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample QR-DQN hyperparameters (QR-DQN = DQN + distributional RL)."""
    additional_args["sample_qrdqn"] = True
    return sample_dqn_params(trial, n_actions, n_envs, additional_args)


def sample_ars_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample ARS hyperparameters."""
    n_delta_pow = trial.suggest_int("n_delta_pow", 2, 6)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
    delta_std = trial.suggest_float("delta_std", 0.01, 0.3)
    top_frac_size = trial.suggest_float("top_frac_size", 0.05, 1.0)

    return convert_ars_params(
        {
            "n_delta_pow": n_delta_pow,
            "learning_rate": learning_rate,
            "delta_std": delta_std,
            "top_frac_size": top_frac_size,
        }
    )


def sample_crossq_params(trial: Any, n_actions: int, n_envs: int, additional_args: dict[str, Any]) -> dict[str, Any]:
    """Sample CrossQ hyperparameters (CrossQ ≈ SAC variant)."""
    return sample_sac_params(trial, n_actions, n_envs, additional_args)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HYPERPARAMS_SAMPLER: dict[str, Any] = {
    "a2c": sample_a2c_params,
    "ars": sample_ars_params,
    "crossq": sample_crossq_params,
    "ddpg": sample_td3_params,
    "dqn": sample_dqn_params,
    "ppo": sample_ppo_params,
    "ppo_lstm": sample_ppo_lstm_params,
    "qrdqn": sample_qrdqn_params,
    "sac": sample_sac_params,
    "tqc": sample_tqc_params,
    "td3": sample_td3_params,
    "trpo": sample_trpo_params,
}

HYPERPARAMS_CONVERTER: dict[str, Any] = {
    "a2c": convert_onpolicy_params,
    "ars": convert_ars_params,
    "crossq": convert_offpolicy_params,
    "ddpg": convert_offpolicy_params,
    "dqn": convert_offpolicy_params,
    "ppo": convert_onpolicy_params,
    "ppo_lstm": convert_onpolicy_params,
    "qrdqn": convert_offpolicy_params,
    "sac": convert_offpolicy_params,
    "tqc": convert_offpolicy_params,
    "td3": convert_offpolicy_params,
    "trpo": convert_onpolicy_params,
}
