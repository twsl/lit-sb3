# lit-sb3

[![Build](https://github.com/twsl/lit-sb3/actions/workflows/build.yaml/badge.svg)](https://github.com/twsl/lit-sb3/actions/workflows/build.yaml)
[![Documentation](https://github.com/twsl/lit-sb3/actions/workflows/docs.yaml/badge.svg)](https://github.com/twsl/lit-sb3/actions/workflows/docs.yaml)

<!--- [![PyPI - Package Version](https://img.shields.io/pypi/v/lit-sb3?logo=pypi&style=flat&color=orange)](https://pypi.org/project/lit-sb3/) -->
<!--- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lit-sb3?logo=pypi&style=flat&color=blue)](https://pypi.org/project/lit-sb3/) -->

[![Docs with MkDocs](https://img.shields.io/badge/MkDocs-docs?style=flat&logo=materialformkdocs&logoColor=white&color=%23526CFE)](https://squidfunk.github.io/mkdocs-material/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/twsl/lit-sb3/releases)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

`lit-sb3` is a [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) adaptation of [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). It preserves the familiar SB3 algorithm API where practical, while moving training orchestration into Lightning so reinforcement learning runs can use Lightning callbacks, loggers, checkpointing, precision settings, and distributed strategies.

## Features

- Stable-Baselines3-style reinforcement learning algorithms implemented on top of PyTorch Lightning
- Lightning-native training through `RLTrainer`, a subclass of `lightning.Trainer`
- Environment management through `RLEnvDataModule`, a `LightningDataModule` for RL stages
- Backwards-friendly `model.learn(total_timesteps=..., env=...)` shim for quick starts
- Support for Lightning callbacks, loggers, checkpointing, mixed precision, and accelerator strategies
- Notebook-friendly workflows, MkDocs documentation, and example notebooks in this repository

## Installation

With `pip`:

```bash
python -m pip install lit-sb3
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv add lit-sb3
```

Optional extras are available for specific workflows and environment families:

```bash
uv add lit-sb3[hub,opt,viz,atari,box2d,classic-control,mujoco,toy-text]
```

## How to use it

Most of the library keeps the familiar sklearn-like style used by Stable-Baselines3.

Here is a quick example of how to train and run PPO on a cartpole environment:

```python
import gymnasium as gym

from lit_sb3 import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO("MlpPolicy", verbose=1)
model.learn(total_timesteps=10_000, env=env) # env needs to be passed here instead of algorithm

vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
```

### Lightning-native workflow

The preferred interface for anything beyond a minimal example is the Lightning-native workflow.

```python
import gymnasium as gym

from lit_sb3 import PPO, RLEnvDataModule, RLTrainer

model = PPO("MlpPolicy")
datamodule = RLEnvDataModule(
    train_env=gym.make("CartPole-v1"),
    test_env=gym.make("CartPole-v1"),
)
trainer = RLTrainer(max_steps=10_000, run_name="ppo_cartpole")

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
```

## Docs

```bash
uv run mkdocs build -f ./mkdocs.yml -d ./_build/
```

## Update template

```bash
copier update --trust -A --vcs-ref=HEAD
```

## Credits

This project was generated with [![🚀 python project template.](https://img.shields.io/badge/python--project--template-%F0%9F%9A%80-brightgreen)](https://github.com/twsl/python-project-template)
