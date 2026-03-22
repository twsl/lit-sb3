"""Minimal lit-sb3 example: PPO on CartPole-v1 and SAC on Pendulum-v1.

This demonstrates that lit-sb3 uses the same API as stable-baselines3,
but the training loop is entirely powered by PyTorch Lightning.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_ppo():
    print("=== PPO on CartPole-v1 ===")
    from lit_sb3.algorithms.ppo.ppo import PPO

    model = PPO("MlpPolicy", verbose=1, n_steps=256, batch_size=64, n_epochs=5)
    model.learn(total_timesteps=5_000, env="CartPole-v1")
    print(f"PPO finished. Total timesteps: {model.num_timesteps}\n")
    return model


def run_sac():
    print("=== SAC on Pendulum-v1 ===")
    from lit_sb3.algorithms.sac.sac import SAC

    model = SAC("MlpPolicy", verbose=1, learning_starts=200)
    model.learn(total_timesteps=2_000, env="Pendulum-v1")
    print(f"SAC finished. Total timesteps: {model.num_timesteps}\n")
    return model


def run_dqn():
    print("=== DQN on CartPole-v1 ===")
    from lit_sb3.algorithms.dqn.dqn import DQN

    model = DQN("MlpPolicy", verbose=1, learning_starts=200)
    model.learn(total_timesteps=2_000, env="CartPole-v1")
    print(f"DQN finished. Total timesteps: {model.num_timesteps}\n")
    return model


if __name__ == "__main__":
    ppo_model = run_ppo()
    sac_model = run_sac()
    dqn_model = run_dqn()

    # Demonstrate SB3-compatible predict API
    print("=== Predict demo ===")
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    action, _ = ppo_model.predict(obs, deterministic=True)
    print(f"PPO action shape: {action.shape}, values: {action}")
    env.close()
