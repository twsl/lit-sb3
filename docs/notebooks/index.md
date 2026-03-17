# Examples

Interactive notebooks demonstrating each algorithm in the `lit-sb3` library.
All examples use the SB3-compatible API backed by PyTorch Lightning.

---

## Introduction

- [Introduction notebook](intro_notebook.ipynb)

---

## On-Policy Algorithms

These algorithms collect a rollout, compute updates, then discard the data.

| Notebook                                           | Algorithm                    | Environment            |
| -------------------------------------------------- | ---------------------------- | ---------------------- |
| [A2C on CartPole-v1](on_policy/a2c_cartpole.ipynb) | Advantage Actor-Critic       | CartPole-v1 (discrete) |
| [PPO on CartPole-v1](on_policy/ppo_cartpole.ipynb) | Proximal Policy Optimization | CartPole-v1 (discrete) |

---

## Off-Policy Algorithms

These algorithms store transitions in a replay buffer and can reuse past experience.

| Notebook                                                              | Algorithm                          | Environment                           |
| --------------------------------------------------------------------- | ---------------------------------- | ------------------------------------- |
| [DQN on CartPole-v1](off_policy/dqn_cartpole.ipynb)                   | Deep Q-Network                     | CartPole-v1 (discrete)                |
| [SAC on LunarLanderContinuous-v2](off_policy/sac_lunarlander.ipynb)   | Soft Actor-Critic                  | LunarLanderContinuous-v2 (continuous) |
| [DDPG on LunarLanderContinuous-v2](off_policy/ddpg_lunarlander.ipynb) | Deep Deterministic Policy Gradient | LunarLanderContinuous-v2 (continuous) |
| [TD3 on LunarLanderContinuous-v2](off_policy/td3_lunarlander.ipynb)   | Twin Delayed DDPG                  | LunarLanderContinuous-v2 (continuous) |
