# Adjustments to the library

The library of the study (SimplyQRL, `third_party/simplyqrl/`, revision `b534cc9`) is vendored **without modification**: a test
(`tests/test_vendored_integrity.py`) checks the checksum of every file. The adjustments that this work needs are applied **when the package
is imported**, by rebinding attributes of the loaded library or by wrapping its objects, in `src/quantumrl_dissection/training/adjustments.py`,
`training/dqn.py` and `environments.py`. Each adjustment is guarded: it checks that the library still has the shape that the adjustment expects and
raises otherwise. The difference between the library and this work is therefore exactly the code listed here. Appendix A of the thesis
describes them; this page says where each one lives and how to check it.

| adjustment | where | purpose | effect on the results |
|---|---|---|---|
| terminal handling | `training/dqn.py` (`SafeDQN`, `AutoresetProbe`, option `terminal_adjustment`) | excludes from the replay buffer the transition produced by the automatic reset of the vectorized environment | changes learning; measured in appendix A.1 (`configs/terminal_ablation.toml`) |
| discrete observations | `environments.py` (`FrozenLake4x4Scalar-v0`, `FrozenLake4x4OneHot-v0`) | presents the state index of FrozenLake as a one-dimensional numerical observation that the Q-network can consume | required for FrozenLake under DQN |
| output scaling | `training/adjustments.py` (`output_scale`) | restores the output-scaling layer along the Q-network branch of the hybrid agent | none: no configuration of the study requests it |
| classical agent type | `training/adjustments.py` (`agent_type_aliases`) | resolves the agent type of the control script of the reference study (`"classic"`) | required to run that control |
| transformer signature | `arms.py` / `circuits/templates.py` (`resolve_transform`) | gives the transformers of FrozenLake the grid size that the code needs and the text of the library omits | none |
| simulator and ceiling agent types | `training/adjustments.py` | registers the agent types of the simulators of the circuit and of the classical models of the study, so that the library builds them | none |
| dependency pins | `pyproject.toml`, `requirements.lock` | pins the dependency set to the lock of the library | required to install the library |

## Terminal handling

Since gymnasium 1.0 the vectorized environment resets on the step that follows a termination. That step ignores the action and returns a
reward of zero and a non-terminal flag together with the reset observation. Stored in a replay buffer, this *phantom transition* teaches
Q(terminal) ≈ max_a Q(s0, a) and is sampled repeatedly. PPO produces the same step but has no buffer in which to keep it.

`AutoresetProbe` wraps `envs.step` and `rb.add` of a constructed DQN. With `terminal_adjustment = true` the phantom transition never reaches the
buffer. In both cases the probe counts the share of such transitions in every run (`terminal_adjustment_share` in `results/runs.csv`,
`outcome.probe` in a manifest), so a run reports the exposure it had even when the adjustment is disabled. The runs with the adjustment
disabled are the arms named `<arm>_no_terminal_adjustment`.

## Seeding of exploration

The agent seeds `random`, `numpy` and `torch`, and the environment is reset with a seed, but the action space of the vectorized environment is
not seeded: the random actions of ε-greedy exploration come from a sampler seeded by the operating system. Two runs with the same nominal
seed can therefore follow different trajectories, mostly early in training. Reproducibility of the main campaign is statistical, not bit for
bit (`docs/reproducibility.md`). In the learning-rate experiment the sampler is seeded with the seed of the run
(`training/lr_equivalence.py`, `make_post_build`), so that two runs can be compared step by step; the library is not touched.

## Selecting components of the observation

The library applies `reuse_indices` (the components of the observation that a replicated input reads) only when the observation is replicated
(`n_repeats > 1`). The reference control without reuse, which reads the components 1, 2 and 3 once, therefore takes them from the environment
(`environments.py`, `SelectObs`; `arms.py`, key `_env_select`): it is a `Linear(3 → 2)` of 8 parameters.
