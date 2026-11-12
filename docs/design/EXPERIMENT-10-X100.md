# Experiment 10 - 100-seed extension on the simulators (design, fixed before any run)

> **SIMULATOR RESULTS - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL**
> ([SIMULATED-RESULTS-POLICY.md](SIMULATED-RESULTS-POLICY.md)). The only approved use of simulator
> results is the validation battery (approval of 2026-11-01); nothing in this experiment is approved.
> Provenance: every manifest carries `provenance.simulated = true`; the result folder carries
> `NOT_FOR_THESIS.md`.

**Written: 2026-11-01, before any cell of this experiment was run.** This document fixes the arms,
the contrasts and the reading rules; they are not changed after seeing data. If something must
change, it is added at the end under a dated "Amendments" heading, never edited above.

## 1. Purpose

Ten seeds leave doubt in several contrasts (for example, P(entangled > unentangled) between 0.64 and
0.92 across Output Reuse values at n = 10). This experiment runs the same contrasts with up to 100
seeds, **on the simulators only** (no real circuit), to see how those contrasts behave with more data
and to decide later what to do with them. It is a source of extra data, not of thesis claims.

## 2. Protocol (the same as the corresponding real experiments)

- CartPole: SimplyQRL's default DQN settings as used by exp01/02/03/07/08: `batch_size` 128,
  `buffer_size` 10 000, `train_frequency` 10, `learning_starts` 10 000 (default), `learning_rate`
  2.5e-4, gamma 0.99, target-network frequency 500, epsilon 1.0 -> 0.05 over the first 50 % of training.
- FrozenLake (Config B, 4 qubits, binary embedding): exp04's protocol: `buffer_size` 50 000,
  `train_frequency` 1, `learning_starts` 1 000, `batch_size` 128; the rest as above.
- FIX-01 on. 100 000 steps. Greedy evaluation every 10 000 steps, 20 episodes. Initial quantum weights
  U[0, 2*pi) (FIX-12).
- Seeds 1-100. Seeds that already exist for an arm **with exactly the same specification** are
  reused (read from where they are, not copied, not re-run); the rest are run with `tag = "x100"` into
  `exp10_emulator_x100/`. "Same specification" = same arm, same `dqn_kwargs`, same
  `total_timesteps`, FIX-01 on, and same tag in the source folder (`""` in exp09, `init2pi` in exp01 for
  `su2_cartpole_L5`). The list of reused seeds is written to `exp10_emulator_x100/reused_seeds.json`.
- Both sides of every contrast go through the **same** simulator, so that any bias of the tool affects
  both sides alike. The `sv` simulator accepts `ent=False`; the arms
  `sv_cartpole_L5_OR{4,8,16,32}_noent` and `sv_cartpole_L{1,2}_noent` are registered as exact twins
  of the corresponding `su2_*` arms (same configuration, `core/configs.py`, section "[exp10]").
- Results live apart: `results/exp10_emulator_x100/`. They are never mixed with exp01-exp09 or with the
  dossier of the thesis.

## 3. Arms by priority

Cells = arm x seeds still to run after reuse (existing 1-10 for sv/su2 CartPole arms; 1-20 for
`sv_frozen_binary_4q_L5`).

| priority | question | arms | cells to run |
|---|---|---|---|
| **P1** | entanglement x Output Reuse, CartPole, L=5 | `sv_cartpole_L5` and `sv_cartpole_L5_OR{4,8,16,32}` (entangled); their `_noent` sv twins (`sv_cartpole_L5_noent`, `sv_cartpole_L5_OR{4,8,16,32}_noent`); secondary check with the SU(2) emulator: `su2_cartpole_L5`, `su2_cartpole_L5_OR{4,8,16,32}` | 1 390 |
| **P2** | depth, CartPole (L=2 -> L=5 has P = 0.63 / 0.70 at n = 10) | `sv_cartpole_L{1,2}` (entangled) and `sv_cartpole_L{1,2}_noent`. L=5 is in P1 | 380 |
| **P3** | depth x Output Reuse, CartPole, L=2 | `sv_cartpole_L2_OR{4,8,16,32}` (entangled). `sv_cartpole_L5_OR32` is in P1 | 360 |
| **P4** (only if time remains) | FrozenLake Config B, L=5: fraction of seeds that solve the task | `sv_frozen_binary_4q_L5` and `sv_frozen_binary_4q_noent_L5` | 170 |

## 4. Contrasts and reading rules (fixed now)

Metrics: `final_perf` and `greedy_final`; IQM with 95 % bootstrap CI per arm; P(a > b) as in the
dossier (unpaired, ties count half).

| id | contrast |
|---|---|
| **C1** | P(entangled > unentangled) at each R in {1, 4, 8, 16, 32}, in `final_perf` and `greedy_final`, with the sv twins (primary); and the same contrast against the su2 arm (secondary) |
| **C2** | P(L2 > L1) and P(L5 > L2), entangled and unentangled |
| **C3** | P(L2_OR{R} > L2) at each R in {4, 8, 16, 32}, and P(L5_OR32 > L2_OR32) |
| **C4** | number of alive seeds (`greedy_final` > 0) out of 100 in each FrozenLake arm, and P(entangled > unentangled) |

Reading rules:
- A contrast is labelled **strong** only if P >= 0.95 in **both** metrics. Otherwise it is reported
  with its numbers and not called strong.
- All contrasts are reported whatever they show. No multiple-comparison correction is applied; that
  is stated in the results.
- Coherence checks: for each arm, seeds 1-10 against seeds 11-100 (they should be indistinguishable);
  and, where it exists, the simulated arm against its real arm at n = 10.
- Nothing is said about when the simulators were used relative to the real experiments in any document
  of this experiment.

## 5. Cost estimate (measured on the existing simulator cells, one thread per cell)

Seconds per cell from the manifests of exp09 and exp01 (mean; 13 workers were running at once):

| arm family | s / cell |
|---|---|
| `sv_cartpole_L5` and its OR variants (8 qubits, L=5) | ~1 050-1 120 |
| `sv_cartpole_L5_noent` | ~1 180 (the OR `_noent` twins are assumed similar: ~1 100) |
| `su2_cartpole_L5` (R=1) / with OR | ~98 / ~210-225 |
| `sv_cartpole_L2` and its OR variants | ~450-580 |
| `sv_cartpole_L1` (R=1) / with OR | ~885 / ~340-450 (the `_noent` twins at L=1, 2 are assumed equal to the entangled ones) |
| `sv_frozen_binary_4q_L5` / `_noent_L5` | ~2 160 / ~1 920 |

| group | cells | core-hours | wall-clock with 14 workers |
|---|---|---|---|
| P1 | 1 390 | ~310 | ~22 h |
| P2 | 380 | ~77 | ~5.5 h |
| P3 | 360 | ~53 | ~3.8 h |
| P4 | 170 | ~96 | ~6.9 h |
| **total** | **2 300** | **~535** | **~38 h** |

Launch order P1, P2, P3, P4; workers claim cells cooperatively. If time runs out, stop at the end of a
whole group, never in the middle of one.

## 6. Outputs

`per_seed_x100.csv` (same columns as the dossier's `per_seed_metrics.csv`), `curves_x100_train_ma50.csv`,
`curves_x100_eval_greedy.csv`, `summary_x100.csv`, and `DOSSIER_EMULADOR_X100.md` (policy notice first,
tables of C1-C4, a one-page reading). Produced by `experiments/exp10_analysis.py` from the manifests.

## Amendments

**2026-11-03 - run incidents (no change to arms, contrasts or reading rules).**

- The sweep ran 2 300 cells (all four groups) with 14 workers. The machine restarted once while 14 cells
  were in flight; those 14 cells (seeds 92 and 93 of the L=5 arms) were run again from scratch by the
  same command and their episode files are consistent (one TensorBoard file per attempt, one episodes CSV).
- In 8 cells of P3 (`sv_cartpole_L2_OR{4,8,16,32}`, seeds 11, 47, 48, 49, 55) two or three workers had
  claimed the same cell at the same moment (race in the cooperative lock), so their episodes CSV could be
  a mix of writers (two of them had a cut line). Those 8 cells were moved, not deleted, to
  `exp10_emulator_x100/_quarantine_dup_writers/` and run again with a single writer. Only the re-runs are
  analysed. An audit of all 2 300 cells (one TensorBoard file per cell; `episode_n` = 1..N; `global_step`
  monotone; last step >= 99 000) finds no other cell with this problem.
- `reused_seeds.json` was overwritten for the four `sv_cartpole_L2_OR*` arms by those re-runs (they were
  launched with `--seeds`); it was regenerated for the full range 1-100 and the script no longer writes
  it when `--seeds` is given.
