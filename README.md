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
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

PyTorch Lightning version of Stable Baselines

## Features

- `...`

## Installation

With `pip`:

```bash
python -m pip install lit-sb3
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv add lit-sb3
```

## How to use it

Most of the code in the library tries to follow the sklearn-like syntax for the Reinforcement Learning algorithms from original Stable Baselines 3.

Here is a quick example of how to train and run PPO on a cartpole environment:

```python
import gymnasium as gym

from lit_sb3 import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO("MlpPolicy", verbose=1)
model.learn(total_timesteps=10_000, env) # env needs to be passed here instead of algorithm

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
```

Or just train a model with a one liner if the environment is registered in Gymnasium and if the policy is registered:

```python
from lit_sb3 import PPO
model = PPO("MlpPolicy")
model.learn(total_timesteps=10_000, env="CartPole-v1")
```

However, the preferred way to use the library is the `RLTrainer` interface, which gives you more control and flexibility over the training loop and allows you to use all the features of `PyTorch Lightning`, such as callbacks, loggers, and more.

```python
import gymnasium as gym
from lit_sb3 import PPO, RLEnvDataModule, RLTrainer

dm = RLEnvDataModule(
    train_env=gym.make("CartPole-v1", render_mode="human"),
    test_env=gym.make("CartPole-v1", render_mode="human")
)
trainer = RLTrainer(max_steps=10_000, rl_callback=PPO(), run_name="ppo_cartpole")

trainer.fit(model=None, datamodule=dm)
trainer.test(model=None, datamodule=dm)
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
