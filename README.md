# QuantumRL-dissection

Code and per-run results of the master's thesis *A Component-Based Evaluation of Parametrized Quantum Circuits: from Proximal Policy
Optimization to Deep Q-Learning Across Configurations and Environments* (Roger Mas Bosch, Universitat Oberta de Catalunya, 2026).

The study takes the component analysis of a hybrid quantum-classical reinforcement-learning agent (output reuse, data reuploading
depth, entanglement), published for PPO on CartPole, and asks which of its findings hold under DQN, on CartPole and on FrozenLake,
against classical controls matched in parameters and in information. It also shows that output reuse followed by a linear layer is
equivalent to multiplying the learning rate of the weights of that layer by R, and tests that prediction. All the experiments use
[SimplyQRL](https://github.com/javier-lazaro/SimplyQRL).

## Installation

Python 3.12. SimplyQRL is installed from its repository at the revision used in the study (`b534cc9`), and the other
dependencies are pinned to exact versions in `pyproject.toml`.

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu    # torch, CPU build
pip install -e ".[dev]"                                                    # adds pytest
```

## Usage

```bash
python experiments/run.py --list                                          # the stages of every experiment and their cells
python experiments/run.py --experiment rq2_capacity --outdir runs --seeds 1-3
python experiments/reproduce.py --out out                                 # the tables and figures from results/
pytest
```

`run.py` writes the episode and evaluation CSVs of every cell to `<outdir>/runs/` and its metrics to `<outdir>/runs.csv`; a cell that is
already there is skipped, so an experiment can be restarted. `--shard I N` splits the cells among N processes. With
`SOURCE_DATE_EPOCH` set, `reproduce.py` writes the PDF files identically from one run to the next.

## Structure

```
src/quantumrl_dissection/
    models.py           circuit configurations, classical controls, classical twin, Fourier ceilings, encodings of the environments
    simulators.py       SU(2) emulator and batched state-vector simulator of the circuits
    arms.py             registry: arm name -> agent type and configuration
    training.py         adjustments to SimplyQRL, DQN with the terminal adjustment, greedy evaluation, one cell
    learning_rate.py    output reuse against the learning rate: variants of DQN, and PPO with a learning rate per group
    analysis.py         statistics, metrics of a run, tables of the study
    appendix_tables.py  tables of the supplementary results
    figures.py          figures
experiments/            experiments.toml (the 12 experiments), run.py, reproduce.py
results/                runs.csv.gz, curves.csv.gz, reference_study.csv.gz
tests/                  test_simulators.py, test_reuse_identity.py
```

`results/runs.csv.gz` has one row per run (6,368): experiment, stage, arm, seed, kind (real circuit, simulator or classical),
training and greedy return, best moving average, area under the moving average, hours of wall-clock time. 5,828 of them are the cells
of `experiments.toml`; the other 540 (variants of the simulator validation, ceilings with other hyperparameters, probes) are in the raw record and count only in the hours.
`curves.csv.gz` holds the learning curves that the figures draw and `reference_study.csv.gz` the logs of the reference study.

## Data

The raw artefacts of every run, the results of the simulators and the supplementary documents (design notes, verification, cost and
energy estimate) are deposited on Zenodo: [10.5281/zenodo.24228739](https://doi.org/10.5281/zenodo.24228739).

## Notes

* **Adjustments to SimplyQRL.** The library is not modified. The adjustments are applied when `quantumrl_dissection` is imported, each
  guarded by a check that the library still has the shape it expects (`training.py`):
  * the transition produced by the automatic reset of the vectorized environment (gymnasium >= 1.0) is not stored in the replay
    buffer (`terminal_adjustment`); the probe counts the share of such transitions in every run;
  * FrozenLake is registered with a numerical observation (`FrozenLake4x4Scalar-v0`, `FrozenLake4x4OneHot-v0`) that the Q-network can consume;
  * the output scale of the hybrid Q-network is appended where it takes effect, and the agent type `"classic"` is resolved;
  * the agent types of the simulators, the classical twin and the ceilings are registered;
  * the components of the observation read by a control without replication are selected in the environment.
* **Reproducibility.** The library does not seed the sampler of the random exploration actions, so two runs with the same seed can
  differ; only the arms of the learning-rate experiment seed it. The results are therefore reproduced statistically, with the
  intervals of the tables. The environment, the hours and the revision of every run are in the data record.
* **Simulators.** The results computed with the simulators of the circuit are additional results; their arms are named `sim_*`.

## Citation and licence

Cite with `CITATION.cff`. The code is MIT licensed (`LICENSE`). `results/runs.csv.gz` and `results/curves.csv.gz` are released
under CC BY 4.0. `results/reference_study.csv.gz` contains the logs of the PPO runs of the reference study (Lázaro et al., 2025),
taken from its repository, which is MIT licensed; the copyright and permission notice is in `LICENSE-reference_study`.
