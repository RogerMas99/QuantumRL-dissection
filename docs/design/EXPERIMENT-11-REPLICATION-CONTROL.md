# Experiment 11 - classical control that replicates its input

Written 2026-11-06, **before any cell of this experiment has run**. Anything that changes afterwards is
added at the end under a dated "Amendments" heading, never edited above.

Real CartPole environment, classical arms only. No simulator is involved.

## 1. Purpose

The thesis concludes that Output Reuse (OR) helps the hybrid agent and not the classical controls. Today
that conclusion rests on two controls, and neither isolates the effect of replication in a classical
network that can learn:

- `classical_OR{R}` (the reference study's control): replicates the input R times, but it is linear (no
  hidden layer), discards the cart position and stays at the floor (~10). A control that does not learn
  cannot show whether replication helps it.
- `classical_or_matched_r{R}`: matches the hybrid's total budget at every R, sees the full observation and
  has a ReLU hidden layer, but its replication is fixed at 4 (`n_repeats = 4`); only the hidden width moves
  with R.

The missing control combines both: replication R + ReLU hidden layer + full observation + budget matched
to the hybrid at every R.

## 2. The new arm

`classical_or_matched_rep_r{R}` (registered in `core/configs.py::build_arm_config`):

- `reuse_indices = [0, 1, 2, 3]` (full observation, `MATCHED_REUSE_INDICES_FAIR`);
- `n_repeats = R`, the same replication as `hybrid_OR{R}`;
- one hidden layer with ReLU;
- width from `match_hidden_width` against the **measured** total of `hybrid_OR{R}`
  (222 / 350 / 606 / 1118 for R = 4 / 8 / 16 / 32), with `in_dim = 4 * R` and `out_dim = 2`.

Widths and parameter counts measured by building each arm and counting trainable parameters:

| R | hybrid_OR total | `classical_or_matched_r` (replication 4): width / params | `classical_or_matched_rep_r` (replication R): width / params |
|---|---|---|---|
| 4 | 222 | 12 / 230 | 12 / 230 |
| 8 | 350 | 18 / 344 | 10 / 352 |
| 16 | 606 | 32 / 610 | 9 / 605 |
| 32 | 1118 | 59 / 1123 | 9 / 1181 |

Because the replicated input makes the first layer consume more of the budget, the hidden width of the new
arm is much smaller than that of `classical_or_matched_r{R}` (10 / 9 / 9 against 18 / 32 / 59 at R = 8 / 16 /
32). That is part of the reading: at equal budget, replication takes capacity away from the hidden layer.
`match_hidden_width` returns the closest width, so the new arm is at most 5.6 % above the hybrid's total
(R = 32: 1181 against 1118) and at most 1 % below.

**R = 4 is not re-run.** With R = 4 the new arm has exactly the specification of the existing
`classical_or_matched_r4` (same indices, `n_repeats = 4`, width 12, ReLU, budget 222; checked by
`tests/test_exp11_arms.py` and by the runner before it starts). Wherever this document says rep_r4 it means
`classical_or_matched_r4` from exp02. Only **R = 8, 16 and 32** are run: 3 arms x 10 seeds = 30 cells.

The name `classical_or_matched_rep_r{R}` starts with `classical_or_matched_r`; the new branch of the arm
builder sits before the old one, and the old one rejects names starting with `classical_or_matched_rep`, so
the old arm can never silently swallow the new name (tested).

## 3. Protocol

Identical to the Output Reuse experiment (exp02): DQN, CartPole-v1, 100 000 steps, `batch_size` 128,
`buffer_size` 10 000, `train_frequency` 10; library defaults for the rest (`learning_starts` 10 000,
`learning_rate` 2.5e-4, gamma 0.99, target network every 500 steps, epsilon 1.0 -> 0.05 over the first 50 %),
terminal fix (FIX-01) **on**, seeds 1-10, greedy evaluation every 10 000 steps. No hyperparameter is tuned:
any negative result is "with these hyperparameters".

**Greedy evaluation size: 5 episodes per checkpoint**, the same as the references of exp02
(`hybrid_OR{R}`, `classical_or_matched_r{R}`), so that `greedy_final` is compared between measurements with the
same noise. exp02 did not pass a number of episodes, so it used the library default of `GreedyEvalConfig`
(5); every greedy score in exp02 is a multiple of 0.2, as a 5-episode mean must be. (Experiments 07-09 used 20;
`docs/` and the thesis text that say "20 episodes" for CartPole in general are wrong for exp02, exp03, exp03b
and part of exp01. The audit is in `handoff_memoria/DOSSIER_v3_ERRATA.md`.) `final_perf` does not depend on this.

Results go to their own folder, `results/exp11_replication_control/`, tag `exp11`, with the usual manifests
and logs. Nothing existing is touched or mixed.

## 4. Metrics and contrasts

Metrics: `final_perf` (mean over the last 10 % of training episodes) and `greedy_final` (mean of the last 3
greedy checkpoints). Per arm: IQM with a 95 % percentile bootstrap CI. Comparisons: P(a > b), unpaired,
ties count 1/2. "Strong" means P >= 0.95 in **both** metrics. **No multiple-comparison correction is
applied** to any contrast.

- **C1. Effect of replication in a classical control at equal budget**, for R in {8, 16, 32}:
  P(`classical_or_matched_rep_r{R}` > `classical_or_matched_r{R}`).
- **C2. The circuit against the control with the same replication**, for R in {4, 8, 16, 32}:
  P(`hybrid_OR{R}` > `classical_or_matched_rep_r{R}`), with rep_r4 = `classical_or_matched_r4`.
- **C3. Trend with R inside the new control:** P(rep_r32 > rep_r4) and P(rep_r16 > rep_r4).
- **C4 (context, no new rule):** the C2 contrasts with the unentangled hybrid,
  P(`hybrid_or_r{R}_noent` > `classical_or_matched_rep_r{R}`).

References used (all existing, FIX-01 on, seeds 1-10): `hybrid_OR{R}` and `classical_or_matched_r{R}` from
exp02, `hybrid_or_r{R}_noent` from exp08.

## 5. Reading rule (fixed before launching)

- If the new control does not improve with replication (C1 and C3 near 0.5) and the hybrid beats it (C2),
  the conclusion that Output Reuse helps the circuit specifically is supported by a control that does learn.
- If the new control improves with replication, Output Reuse also acts as a generic optimisation mechanism
  in classical networks, and the conclusion must be qualified.
- If the new control does not learn at all, the contrast is again uninformative, and it is reported as such.

**Operationalisation of the three conditions** (our choice, made here, before any cell runs; it does not
change the rule above, it makes it checkable and `exp11_analysis.py` applies it mechanically):

- "near 0.5": P in [0.25, 0.75] in both metrics (the same band the project uses for its coherence checks).
- "improves": strong, P >= 0.95 in both metrics.
- **Intermediate values** (P between 0.75 and 0.95, or between 0.05 and 0.25, i.e. neither strong nor near
  0.5): reported as "indicates a direction", as in the rest of the thesis, with the direction stated; if the two
  metrics point in opposite directions, as "mixed". There is no gap without a rule.
- "hybrid beats it": strong in C2 (P >= 0.95 in both metrics).
- "does not learn": IQM `final_perf` <= 12 and IQM `greedy_final` <= 12 at every R in {4, 8, 16, 32} (the
  project's threshold for an informative CartPole pair), with rep_r4 = `classical_or_matched_r4`.
  **The R = 4 reference sits right at that limit** (exp02: IQM `final_perf` 12.4, `greedy_final` 13.8,
  marginally above 12). The dossier will state its values and that it lies at the limit, instead of classifying
  it silently.

**Budget overshoot.** The R = 32 arm has 1181 parameters against the hybrid's 1118 (+5.6 %). That goes in the
direction that favours the control (it gets slightly more than the hybrid), the same rule used for the
FrozenLake matched control, which rounds up rather than down. It is accepted and noted.

## 6. Cost

Classical networks of 230-1181 parameters: about 50 s per cell in the existing matched-control cells.
30 cells, a few minutes of wall clock with a handful of workers.

## 7. Outputs

`per_seed_exp11.csv` (same columns as `per_seed_metrics.csv` of the dossier), `curves_exp11_train_ma50.csv`,
`curves_exp11_eval_greedy.csv`, `summary_exp11.csv`, `tables_exp11.md` and `DOSSIER_EXP11.md` (design,
measured widths and parameters, IQM per arm, tables C1-C4 with P in both metrics, application of the reading
rule, answer to the parameter-count check). Produced by `experiments/exp11_replication_control.py` (runner)
and `experiments/exp11_analysis.py`.

## Amendments

**2026-11-06 - run record (no change to arms, contrasts or reading rule).** The 30 cells (R = 8, 16, 32 x seeds
1-10) ran with 5 workers, each with its own pair of seeds and no cooperative locks, in about 4 minutes. No
incident: one TensorBoard file per cell, no failed cell. The author confirmed 5 episodes for the greedy
evaluation, the two additions to the reading rule (intermediate values as "indicates a direction"; the R = 4
reference stated explicitly at the limit of the 12 threshold) and the +5.6 % budget at R = 32 before launch.
