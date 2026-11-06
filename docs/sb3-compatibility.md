# Stable-Baselines3 Compatibility

Comparison target: Stable-Baselines3 2.9.0, the linked upstream project's current release at the time of this comparison. This page distinguishes SB3 behavior retained by `lit-sb3` from changes made for Lightning integration. It is not a claim of drop-in import or checkpoint compatibility.

## Compatibility Matrix

| Surface | `lit-sb3` status | Difference or coverage |
| --- | --- | --- |
| Core algorithms | A2C, ARS, DDPG, DQN, PPO, SAC, TD3, and HER replay buffer are present. | QRDQN, TQC, CrossQ, MaskablePPO, RecurrentPPO, and TRPO are additional extension algorithms, not part of the linked SB3 core package. |
| Policies and prediction | SB3-style policy aliases and `predict(observation, state, episode_start, deterministic)` are provided. | Import paths use `lit_sb3`; `stable_baselines3` is not an alias namespace. |
| Training | `learn(total_timesteps, ...)` remains available and delegates training to Lightning. | Environment can be passed to `learn(env=...)`; SB3 normally binds it in the algorithm constructor. `log_every_n_steps` is used for Lightning logging instead of SB3's `log_interval`. |
| Environment ownership | `get_env()` and `set_env()` are present. | The active environment belongs to `RLTrainer`/`RLEnvDataModule`; `set_env()` requires an attached `RLTrainer`, unlike standalone SB3 models. |
| Vector environments | The legacy SB3 VecEnv contract is implemented: reset returns observations, step returns `(obs, rewards, dones, infos)`, completed episodes autoreset, and terminal observations/time-limit flags are recorded. | Deterministic contract assertions are in `tests/unit/test_vec_envs.py`; broader wrapper and subprocess behavior remains covered by existing unit tests. |
| Buffers | Replay, rollout, dict, recurrent, maskable, and n-step buffers are present. | Existing tests cover shapes, dtypes, devices, and normalization; timeout and sample-by-sample differential comparisons are incomplete. |
| Serialization | Model `save()`/`load()` and parameter APIs are present. | Cross-library archive interchange is not promised. A deterministic local round-trip test is in `tests/integration/test_on_policy.py`. |
| Lightning integration | `RLTrainer` and `RLEnvDataModule` support Lightning training and evaluation stages. | This is an additional execution path, not an SB3 API equivalent. Lightning module checkpoints and SB3 ZIP archives have different ownership and serialization semantics. |

## Test Status

The new vector-environment test checks reset infos, autoreset observations, `terminal_observation`, and `TimeLimit.truncated` for both termination and truncation. The checkpoint round-trip test checks deterministic action preservation and timestep restoration. Existing buffer tests are adapted from SB3 tests, and integration tests verify representative algorithm training workflows.

These tests assert versioned SB3 contracts using local deterministic fixtures; they do not yet instantiate SB3 2.9.0 as a runtime oracle. Exact loss/target parity across every core algorithm, the complete model API matrix, and parity against upstream wrappers and buffer samples remain open. Do not interpret training-return similarity as a parity guarantee: results are stochastic and depend on framework, device, and runtime versions.

## References

- [SB3 2.9.0 documentation](https://stable-baselines3.readthedocs.io/en/v2.9.0/)
- [SB3 2.9.0 source](https://github.com/DLR-RM/stable-baselines3/tree/v2.9.0/stable_baselines3)
- [SB3 2.9.0 tests](https://github.com/DLR-RM/stable-baselines3/tree/v2.9.0/tests)