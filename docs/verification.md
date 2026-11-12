# Verification

This page reports what was checked to support the statement that the code of this repository computes what the code that produced the runs computed, with the
figures of each check. The tables are in `results/verification/`; `scripts/verify_equivalence.py` repeats the checks given the original repository.

## Method

The code of the original repository is exported with `git archive` at the revision that produced each run (`effective_revision` of `results/runs.csv` when it is set, otherwise
`code_revision`; for the 140 runs that record no revision, `41005cc` if the run started before the library was added to the original repository and `18e124a` after, see
`docs/reproducibility.md`). Both codes run in the same environment (Python 3.11, torch 2.13.0 on CPU, PennyLane and Lightning 0.41.1), one thread, **double precision**, and the
random actions of the exploration are drawn from a generator that the harness seeds with the same seed on both sides. Everything else, from the initialization of the weights to the
replay buffer, the updates and the optimizer, is what each code does. For every arm the configuration of the arm is taken from the configuration file of this repository on the new
side and from the registry (or the helper that the first scripts called) of the original code at that revision on the old side.

## Results

| check | items | result |
|---|---|---|
| Parameters: number of trainable parameters and name and shape of every tensor of the online network | 335 pairs (arm, revision): 219 combinations of configuration, stage and arm, 168 arm names, 29 revisions of the original code | identical in all (`params.csv`) |
| Traces: the same actions and weights after 2,000 steps (updates from step 200), with the larger of the weight differences at most 1e-12 | 62 items: 44 arms of every type, each at its first and at its last revision | the 2,001 actions are the same in all and the largest weight difference is 0.0 in all (`traces.csv`); the weights move by at least 0.049 during the run, so the updates are exercised |
| Traces of PPO, float64, 4,096 steps (8 updates) | the 20 PPO arms | identical weights, difference 0.0 (`ppo.csv`) |
| Full run: 100,000 steps with the protocol of the configuration | the reference control with reuse 8 (classical) and the one-qubit circuit with depth 1 ported from the reference study (real circuit) | the 100,001 actions are the same and the weights differ by 0.0 (`full.csv`) |
| Registry: configuration of every arm field by field | the 176 arms of the original registry at its last revision | identical, with the names translated (`tests/test_arm_registry.py`, `tests/data/arm_configs.json`) |

The arms of the traces cover: the hybrid agent with and without entanglement, with reuse and with depth; the arms ported from the reference study (Skolik, Salinas, Hsiao);
the FrozenLake circuits (binary and scalar), their ceilings, the one-hot network and the matched control; the reference control, the matched and large networks, the classical twin
(bounded and unbounded), the Fourier ceilings, the networks without replication and the networks on the bits, with the learning-rate arms (head, collapsed start, gradient, global);
the terminal-handling ablation; and the SU(2) emulator and the state-vector simulator, in CartPole and in FrozenLake.

Two checks that were run while the repository was built, from the data record that is not part of the repository, and that therefore cannot be repeated from here:

* The configuration recorded in the manifest of each run, compared field by field with the configuration that the registry gives to its arm: 879 of the 927 distinct configurations
  coincide. The 48 that do not are 7 of the first campaign whose manifests record no configuration and 41 of the learning-rate experiment, whose keys had a different name in the
  original code and whose arms behave identically in the traces above.
* The metrics of every run: the 830 per-seed rows of the tables of the thesis are reproduced exactly from `results/`.

## What these checks do not show

* They show that, on one machine, the two codes execute the same computation. They do not show that a run of the study would give the same result today: the runs were produced on
  Colab and on a personal computer, the exploration sampler of the library is not seeded (`docs/adjustments.md`), and a run repeated with the same nominal seed is not bit-identical
  (the same holds for the repetition of one run of the reference control made as a check).
* The traces are 2,000 steps long (PPO: 4,096) except for the two full runs: a difference that only appears later, or in an arm and a revision that were not traced, would not show.
  The parameter check covers every arm and revision.
* For the 140 runs without a recorded revision, the revision is inferred from the date; for the groups of `docs/reproducibility.md` that were launched from a working tree, the
  state of the tree between the start of the run and its commit is not recorded.
* One ceiling of FrozenLake (the linear one on the binary encoding) builds single-precision buffers in both codes and was traced in single precision; the arm of the terminal-handling
  ablation and the arms of the pilot take the name of the arm that they repeat.
* The comparison with the thesis (`make_tables.py --pdf`) is a check of presence: it looks each number up in the text of the page. Of the tables, E.5 and F.1 differ in one number each
  (an interval of 0.67 to 0.80 against 0.66 to 0.80, and 0.5 hours of simulator time against 0.6; the cell of F.1 is 0.547 hours, the 20 runs of the SU(2) emulator, discarded ones included,
  see `docs/run_reconciliation.md`), and table 4.3 prints only part of the ranges that are computed (the rest is in table E.1, which matches).
* Figures 4.5 and 4.7 were also compared with the vector paths of the printed figures. In figure 4.5 the central lines (IQM of the 50-episode moving average) coincide to at most 0.06 units of
  return in both panels, and the bands are the 95% percentile bootstrap interval of the IQM, as the caption states; the bands of the printed figure and of this repository differ by at most
  3.5 units at any point, which is the size of the resampling noise of a bootstrap of 4,000 resamples with ten seeds (the seed of the printed one is not recorded). The style (colours, transparency
  and width of the lines) follows the printed figure. In figure 4.7 the connectors and the shaded bands take the colours of the printed figure; the labels of the research questions are placed differently.

## Repeat

```bash
python scripts/verify_equivalence.py params  --old-repo <original repository> --workdir work
python scripts/verify_equivalence.py traces  --old-repo <original repository> --workdir work --steps 2000
python scripts/verify_equivalence.py ppo     --old-repo <original repository> --workdir work
python scripts/verify_equivalence.py full    --old-repo <original repository> --workdir work/full_a --config rq3_output_reuse --stage reuse_sweep --arm reference_control_reuse8
python scripts/verify_equivalence.py export  --workdir work
```

`traces` takes about an hour and a half on a personal computer with `--jobs 7`, `params` about half an hour, and each full run between one and several hours (the real circuit is slower).
