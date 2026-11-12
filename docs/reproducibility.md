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
* **The learning-rate experiment was launched from a working tree that was committed afterwards.** The 2,130 manifests of that experiment record the revision
  `339f2a4` of the original repository, a commit that contains only a README; the code that produced the runs was committed in `3be15be` the next day. The
  code of this repository corresponds to `3be15be`.
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
