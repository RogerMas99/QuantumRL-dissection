# Reproducibility

This page says what can be reproduced from this repository, how, and what cannot. It states the limits without softening them.

## What the repository contains and what it does not

* **Code** that produced the results, reorganized and renamed: `src/quantumrl_dissection/`, `configs/`, `scripts/`, `tests/`. The vendored library is in
  `third_party/simplyqrl/` (revision `b534cc9`, unmodified). The reorganization changed names and layout, not behaviour: the arm registry reproduces the
  configuration of every arm of the campaign field by field (`tests/test_arm_registry.py`), the metrics reproduce the per-seed values of the original
  tables, and the equivalence of the training code with the code that produced the runs was checked step by step (see "Verification" below).
* **Results** (`results/`, CC BY 4.0): one row per run in `results/runs.csv` (6,368 runs), and per configuration the training episodes, the greedy
  evaluations and, in the learning-rate experiment, the per-1,000-step diagnostics (`results/<config>/*.csv.gz`). Every table and figure of the study is
  recomputed from them.
* **Not in the repository**, but in the data record of the study: the raw artefacts of each run (manifests, TensorBoard logs, Q-value traces, weights,
  the double-precision traces), the runs of the exploratory groups of the simulator sweep that the study does not use, and the calibration of the
  energy measurement. They are not needed to regenerate any table or figure.

## Recomputing the tables and figures

```bash
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
python scripts/make_tables.py  --out tables  --pdf thesis.pdf     # all result tables; with the PDF, every number is looked up in the printed table
python scripts/make_figures.py --out figures                      # figures 4.1 to 4.7
```

The tables are computed from the data; the comparison with the PDF only reports. Where a number differs it is reported as it is: `tables/comparison_report.txt`.

## Running an experiment again

```bash
python scripts/run.py --config configs/rq3_output_reuse.toml --list              # the cells of the configuration
python scripts/run.py --config configs/rq3_output_reuse.toml --outdir runs/ --seeds 1-3
```

Each configuration lists, per stage, the arms, seeds, steps, DQN hyperparameters, the terminal adjustment and the evaluation protocol of the runs of the
study; they were generated from the manifests of those runs. Runs are resumable and a finished cell is skipped.
On Colab: `pip install` the repository in the session, mount a folder for `--outdir` and run the same commands; use `--claim` when several sessions share a folder.

Cost: the real circuit on CartPole took about 9 hours per run of 100,000 steps (hybrid agent) on Colab, and about 25 hours per run of the binary circuit on FrozenLake on
the personal computer; the classical arms take seconds to minutes. Table F.1 of the thesis, recomputed by `make_tables.py`, gives the totals.

## Reproducibility of a run

Training is reproducible **statistically, not bit for bit**. The initialization of the weights and the sampling of the replay buffer are seeded and ties in
the maximum are broken deterministically, but the random actions of ε-greedy exploration come from a sampler that the library does not seed
(`docs/adjustments.md`, "Seeding of exploration"). Two runs with the same nominal seed can follow different trajectories. Every result is an interval estimate over
independent seeds. **Exception:** in the learning-rate experiment the sampler is seeded with the seed of the run, so that two runs can be compared step by step; this
is what the double-precision comparisons of that experiment use.

## What is not recorded

* **Revision of the code of 150 runs.** The manifests of 140 runs of the first campaign (the reuse sweep: 80; the depth sweep: 30; the depth sweep without
  entanglement: 30) do not record the revision of the code that produced them; neither do the 10 runs of the reference control without reuse (added afterwards; their
  code is the revision `d0aa338` of the original repository). In `results/runs.csv` the column `code_revision` is empty for these 150 runs.
* **Five groups of runs were launched from a working tree that was committed afterwards.** The manifest records the revision that was checked out when the run
  started; the code that ran was in the working tree and was committed later, as the revision in the last column. `results/runs.csv` keeps the recorded revision in
  `code_revision` and gives the revision that holds the code in `effective_revision`, with the reason in `revision_note`.

  | runs | recorded revision | revision that holds the code | runs started (UTC) | commit (UTC) |
  |---|---|---|---|---|
  | 10 runs of the emulator without entanglement used as the reference of the validation (`emulator_reference`; also the first ten seeds of the 100-seed comparison) | `82bb533` | `5ae1011` | 2026-10-20 04:11 to 04:26 | 2026-10-20 04:32 |
  | 30 runs of the classical controls that replicate their input (`rq3_output_reuse`) | `0524a76` | `7e2210f` | 2026-11-06 14:23 to 14:26 | 2026-11-06 14:34 |
  | 100 runs of the classical twin (`rq2_capacity`) | `d80697f` | `6aa668a` | 2026-11-06 15:59 to 16:09 | 2026-11-06 16:11 |
  | 2,130 runs of the learning-rate experiment | `339f2a4` | `3be15be` | 2026-11-06 19:44 to 2026-11-07 13:07 | 2026-11-07 14:42 |
  | 10 runs of the reference control without reuse | none | `d0aa338` | 2026-11-08 10:25 to 10:28 | 2026-11-08 10:29 |

  In the first group the difference matters: at `82bb533` the emulator initializes its weights at zero, in `5ae1011` uniformly in [0, 2π) as the real circuit does;
  the code of this repository does the latter. The final weights saved by those ten runs are distributed over [0, 2π) (pooled mean 3.25, standard deviation 1.84;
  uniform: 3.14 and 1.81), whereas the runs that started from zero (recorded revision `961548b`, kept in the table with `in_results = false`) end with all their
  weights at exactly zero. The state of the working tree between the start of a run and its commit is not recorded, so that the code that ran cannot be proved identical
  to the committed code. For the files that the last four groups introduced (the classical twin, the modules of the learning-rate experiment) the committed version is
  the only one in the history.
* **Dating of the runs.** `started_utc` in `results/runs.csv` is the time stamp in the name of the TensorBoard log of the run (the last log when a run has several,
  which are interrupted attempts: 281 of the directories of logs have more than one). For each run the commits that touch the modules that the arm runs on were compared with
  that date. Apart from the first group above, every such commit that was made after a run started, or that is contained in the recorded revision but was made after the run started
  (31 runs: 9 of the FrozenLake experiment, up to 24 hours before the commit of their recorded revision, and 22 of the simulator validation, minutes before it), only adds
  arms, options or files: no behaviour of an existing arm changes (`revision_note` says so for those runs). Before the vendored copy of the library was added to the repository
  (2026-08-05), the 27 runs without a recorded revision that started earlier (reuse sweep and depth sweep, from 2026-07-28) ran with a copy of SimplyQRL whose revision is not
  recorded. The messages of the four commits of the upstream repository after 2025-06-25 (up to `b534cc9`, 2025-10-31) describe documentation, licence and citation changes, but
  the revision that those runs used is not determinable.
* **The hardware of the runs on Colab** (processor, accelerator, session) and the **versions of the software on Colab** (the manifests store only the version of Python:
  3.12 and 3.13 on Colab, 3.11.15 on the personal computer). `results/runs.csv` records, in `log_origin`, where each run wrote its logs (`colab` or `local`), which
  indicates where it ran without proving it. The personal computer was an AMD Ryzen AI 7 350 (8 cores, 16 threads), 31 GB, CPU only.
* **The seeds that were reused** when a grid was resumed, and the time of runs that were interrupted, repeated or overwritten, are not recorded and are not estimated.
* **Revisions inside the same experiment.** The runs of one experiment of the first campaign were produced at several revisions of the code (up to 14 in the
  FrozenLake experiment); `results/runs.csv` records the revision of each run.

## Runs and totals

`results/runs.csv` has 6,368 rows: the 6,366 runs of the campaign with a manifest (including the pilot of 60,000 steps and the 10 runs of the reference control without
reuse) and the two test runs of the terminal-handling probe. The 8 duplicated runs that the original repository kept apart, and the repetition of one run made to check
reproducibility, are not in the table. The runs of the exploratory groups of the simulator sweep and the runs superseded by a correction of the emulator, which the study does not
use, are in the table with `in_results = false`: their curves are in the data record, not in `results/`. The table `docs/run_reconciliation.md` gives the count by configuration and
reconciles it with the 6,368 runs of table F.1 of the thesis.

## Verification

The statements of this page about the equivalence between this code and the code that produced the runs are backed by `docs/verification.md`, which reports the checks and their figures.
