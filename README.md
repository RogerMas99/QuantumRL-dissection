# QuantumRL-dissection

Code, configurations and per-run results of the master's thesis **A Component-Based Evaluation of Parametrized Quantum Circuits: from Proximal Policy
Optimization to Deep Q-Learning Across Configurations and Environments** (Roger Mas Bosch, Universitat Oberta de Catalunya, 2026).

The study takes the component analysis of a hybrid quantum-classical reinforcement-learning agent (output reuse, data reuploading depth, entanglement), published for
PPO on CartPole, and asks which of its findings hold under DQN, on CartPole and on FrozenLake, against classical controls matched in parameters and in information, and
against classical models that represent the same class of functions as the circuit. It also shows that output reuse followed by a linear layer is a multiplication of the learning rate
of that layer, and tests that prediction. All experiments use [SimplyQRL](https://github.com/javier-lazaro/SimplyQRL), vendored here without modification.

> The previous repository, `RogerMas99/qrl-dissection`, is archived and public: it holds the history of the work. This repository is the clean version of the code, with the names of the thesis.

## Contents

```
src/quantumrl_dissection/   circuits/ agents/ simulators/ training/ analysis/ arms.py (the registry: arm name -> configuration)
third_party/simplyqrl/      the library, unmodified (revision b534cc9), with its MIT licence
configs/                    one file per research question (and the learning-rate experiment, the terminal ablation, the simulators)
scripts/                    run.py (run a configuration) · make_tables.py · make_figures.py · verify_equivalence.py
results/                    runs.csv (6,368 runs) and, per configuration, episodes, greedy evaluations, diagnostics; results/reference_study/
docs/                       design/ (original design documents) · adjustments.md · reproducibility.md · arm_names.csv · verification.md
tests/                      the test suite
```

## Install

Python 3.11. The exact versions of the environment of the study are in `requirements.lock` (PennyLane and Lightning 0.41.1, gymnasium 1.1.1, torch 2.13.0 on CPU, ...).

```bash
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
python -m pytest tests            # about 4 minutes
```

## Reproduce each table and figure

```bash
python scripts/make_tables.py  --out tables  [--pdf thesis.pdf]    # every result table, recomputed from results/
python scripts/make_figures.py --out figures                       # figures 4.1 to 4.7
```

With the PDF of the thesis, `make_tables.py` looks up every number of each table in the printed table and reports the ones that it does not find (`tables/comparison_report.txt`).
Nothing is adjusted: a difference is reported. The statistics are the IQM, the 95 % percentile bootstrap interval (10,000 resamples, `default_rng(0)`) and the unpaired
probability of improvement P(a > b) (ties count one half); "strong" means P ≥ 0.95 in both statistics and "level" P in [0.35, 0.65] in both.

| section of the thesis | tables and figures | configuration | results |
|---|---|---|---|
| RQ1, porting the reference configurations (3.6.1, 4.1) | 4.1, D.6, E.4 | `rq1_porting` | `results/rq1_porting/` |
| RQ2, capacity ladder and classical twin (3.6.2, 4.2) | 4.2, D.2, D.3, fig. 4.1 | `rq2_capacity` (and `lr_experiment` for the network without replication) | `results/rq2_capacity/` |
| RQ3, output reuse (4.3.1) | D.4, fig. 4.2, 4.3 | `rq3_output_reuse` | `results/rq3_output_reuse/` |
| RQ3, output reuse as a scaling of the learning rate (4.3.2, 3.6.3) | 4.3, E.1, E.2, E.3, fig. 4.4 | `lr_experiment` | `results/lr_experiment/` |
| RQ3, reuploading depth and its crossing with reuse (4.3.3, 4.3.5) | 4.5, D.5, fig. 4.5 | `rq3_depth` | `results/rq3_depth/` |
| RQ3, entanglement (4.3.4) | 4.4 | `rq3_entanglement` | `results/rq3_entanglement/` |
| RQ4, Fourier ceilings (4.4) | 4.6 | `rq4_ceilings` | `results/rq4_ceilings/` |
| RQ5, FrozenLake (4.5) | 4.7, fig. 4.6 | `rq5_frozenlake` | `results/rq5_frozenlake/` |
| key contrasts and synthesis (4.7) | fig. 4.7 | all of the above | |
| terminal handling (appendix A.1) | A.2 | `terminal_ablation` | `results/terminal_ablation/` |
| validation of the simulators (appendix C.5) | C.1 | `simulator_validation` | `results/simulator_validation/` |
| additional results with 100 seeds (appendix E.3) | E.5 | `additional_x100` | `results/additional_x100/` |
| cost (appendix F) | F.1, F.2 | | `results/runs.csv` |

The results computed with the **simulators** of the circuit (validation, 100 seeds, and the hybrid agent in the learning-rate experiment) are additional results, kept apart from those of
the real circuit; their arms are named `sim_*`.

## Run an experiment

```bash
python scripts/run.py --config configs/rq3_output_reuse.toml --list              # the cells
python scripts/run.py --config configs/rq3_output_reuse.toml --outdir runs/ --seeds 1-3
```

On Colab: install the repository in the session (`pip install -r requirements.lock` and `pip install -e .`), mount a folder for `--outdir`, and run the same commands; with
several sessions on the same folder add `--claim`. Hardware and times: the runs of the real circuit are slow (about 9 hours per run of 100,000 steps for the hybrid agent on CartPole,
about 25 hours for the binary circuit on FrozenLake), the batched simulator runs at 60 to 80 steps per second, the SU(2) emulator at 170 to 430 and the classical arms at 900 to 2,000
(personal computer, AMD Ryzen AI 7 350, 31 GB, CPU only). The total is about 4,127 hours of wall-clock time summed over runs (table F.1); it is not calendar time, because runs ran in parallel. `docs/reproducibility.md` states what is not recorded.

## Arm names

The name of an arm describes its specification; the experiment is the `config` of the run. `docs/arm_names.csv` translates the names that the original repository used, and lists the arms
that were merged because their configuration is identical.

## Third-party code and data

* `third_party/simplyqrl/`: SimplyQRL, Copyright (c) 2025 Javier Lazaro Gonzalez, MIT (`third_party/simplyqrl/LICENSE`), at revision `b534cc9`, byte for byte (checked by `tests/test_vendored_integrity.py`).
  The adjustments that this work applies to it are in `src/quantumrl_dissection/training/adjustments.py` and are described in `docs/adjustments.md`.
* `results/reference_study/`: the per-seed logs of the PPO runs of the reference study (Lazaro et al.), taken from its repository, MIT licensed.

## Licence and citation

Code: MIT (`LICENSE`). `results/`: CC BY 4.0 (`NOTICE-results.md`). Cite with `CITATION.cff`; the data record of the study (raw artefacts of every run) is deposited separately.
