# Experiment 12 - classical twin of the hybrid agent

Written 2026-11-06, **before any cell of this experiment has run**. Anything that changes afterwards is added at
the end under a dated "Amendments" heading, never edited above.

Real CartPole environment, classical arms only. No simulator is involved.

## 1. Purpose

Output Reuse (OR) acts in the hybrid agent only at the border between the circuit and the linear head: the circuit
stays fixed at 80 parameters and the only thing that grows with R is the head. A faithful classical control is
therefore a **twin**: the same head and the same OR in the same position, with the circuit replaced by a small
classical extractor with the same number of outputs and the same range. The only thing that then differs between
twin and hybrid is where the 8 features come from.

## 2. Architecture of the hybrid, checked in the code

`hybrid_fig4` / `hybrid_OR{R}` (`src/simplyqrl/agents.py`, `HybridAgent` with `is_qnet=True`, config in
`core/configs.py`):

    obs (4, not transformed) -> PQC Skolik, 8 qubits, L = 5, entangled (80 weights) -> 8 values <Z_i> in [-1, 1]
      -> OutputReuse(R) (x.repeat(1, R): 8R values) -> Linear(8R -> 4) -> Identity -> Linear(4 -> 2)

The linear layers use the default `nn.Linear` initialisation (no `layer_init`); one Adam optimiser over all
parameters (`simplyqrl/dqn.py`). Head: 32R + 4 + 10 parameters (46 for R = 1, 142 for R = 4). Measured totals of
the hybrid: 126 / 222 / 350 / 606 / 1118 for R = 1 / 4 / 8 / 16 / 32. All of this was confirmed by building the
agents (`tests/test_exp12_arms.py`).

## 3. The arms

Two families, R in {1, 4, 8, 16, 32} (with R = 1 there is no OutputReuse, as in `hybrid_fig4`):

- **A. `classical_twin_r{R}`** (main variant, bounded outputs like the circuit):
  `obs (4, not transformed, full observation) -> Linear(4 -> 6) -> ReLU -> Linear(6 -> 8) -> Tanh`
  (86 parameters: 30 + 56) `-> OutputReuse(R)` (the very module the hybrid uses) `-> Linear(8R -> 4) -> Identity
  -> Linear(4 -> 2)` (the hybrid's head).
- **B. `classical_twin_unbounded_r{R}`** (range control): the same without the final Tanh.

The extractor layers use the default `nn.Linear` initialisation, like the head (the order does not specify one; this
is the library default and nothing is tuned).

### How it is built, and why the result is equivalent

`core/classical_twin.py::ClassicalTwinAgent` **is a** upstream `HybridAgent`. It calls the upstream constructor with
the config of `hybrid_OR{R}` (`hybrid_or_config(R)`, plus two keys `twin_hidden = 6`, `twin_bounded`), so OutputReuse,
the head layers, their default initialisation, the activation and `forward` are the upstream objects and the upstream
code, not a copy; then it replaces the first module of `self.network` (the PennyLane layer) by the classical
extractor and drops the discarded layer. The vendored library is not modified. Consequences, all tested:

- head and OR have the same classes and shapes as `hybrid_OR{R}` (same layers, same sizes, same `n_repeats`);
- under one seed the head starts with exactly the weights of the hybrid's head (the head is built by the same code
  from the same RNG state, before the extractor draws);
- no PennyLane layer and no duplicate registration is left in the twin; the optimiser and the DQN loop see an ordinary
  `HybridAgent`.

Reading note: because the twin's config is the config of `hybrid_OR{R}` plus the two twin keys, its manifests
record `circ_type: skolik`, `ent: True`, `n_layers_q: 5` in `agent_config`. Those keys only define the head
(`n_qubits`, `reuse_repetitions`, `net_arch`, `activation`); no circuit exists in the twin. What identifies the arm is
`agent_type: classical_twin` and the `twin_*` keys.

The arms are registered in `core/configs.py` as exact keys of `ARMS` (agent type `classical_twin`, dispatched in
`core/compat.py`); no prefix matching is involved, and `classical_twin_r` cannot capture `classical_twin_unbounded_r`
or any `classical_or_matched_*` arm (tested).

### Parameters (measured by building each arm)

| R | hybrid total | twin total (A and B) | extra vs hybrid | head | extractor |
|---|---|---|---|---|---|
| 1 | 126 | 132 | +6 | 46 | 86 |
| 4 | 222 | 228 | +6 | 142 | 86 |
| 8 | 350 | 356 | +6 | 270 | 86 |
| 16 | 606 | 612 | +6 | 526 | 86 |
| 32 | 1118 | 1124 | +6 | 1038 | 86 |

The twin has 6 more parameters than the hybrid at every R (the extractor, 86, against the circuit, 80). That excess
goes in the direction that favours the control; it is accepted and noted.

## 4. Protocol

Identical to the Output Reuse experiment (exp02) and to `hybrid_fig4` of exp01: DQN, CartPole-v1, 100 000 steps,
`batch_size` 128, `buffer_size` 10 000, `train_frequency` 10; library defaults for the rest (`learning_starts`
10 000, `learning_rate` 2.5e-4, gamma 0.99, target network every 500 steps, epsilon 1.0 -> 0.05 over the first
50 %), terminal fix (FIX-01) **on**, seeds 1-10. **Greedy evaluation every 10 000 steps with 5 episodes**, passed
explicitly, the same as `hybrid_fig4` (exp01) and `hybrid_OR{R}` (exp02), so that `greedy_final` has the same noise
as its references. No hyperparameter is tuned: any negative result is "with these hyperparameters".

2 families x 5 values of R x 10 seeds = **100 cells**. Own folder `results/exp12_classical_twin/`, tag `exp12`, with
the usual manifests and logs. Nothing existing is touched or mixed.

## 5. Metrics and contrasts

Metrics: `final_perf` (mean over the last 10 % of training episodes) and `greedy_final` (mean of the last 3 greedy
checkpoints). Per arm: IQM with a 95 % percentile bootstrap CI. Comparisons: P(a > b), unpaired, ties count 1/2.
**No multiple-comparison correction is applied to any contrast.** "Strong" means P >= 0.95 in **both** metrics;
"near 0.5" means P in [0.25, 0.75] in both; intermediate values are "indicates a direction" (with its sense);
"mixed" if the two metrics point in opposite directions. References (existing, FIX-01 on, seeds 1-10):
`hybrid_fig4` (exp01) for R = 1 and `hybrid_OR{R}` (exp02) for R >= 4.

- **G1. Effect of OR in the twin:** P(`twin_r{R}` > `twin_r1`) for R in {4, 8, 16, 32}.
- **G2. Effect of OR in the hybrid, same metric, as reference:** P(`hybrid_OR{R}` > `hybrid_fig4`) (recomputed with
  the same scripts, so it is comparable).
- **G3. Circuit against twin at equal R:** P(`hybrid_OR{R}` > `twin_r{R}`) for R in {1, 4, 8, 16, 32}
  (`hybrid_fig4` at R = 1).
- **G4. Role of the bounded range:** P(`twin_r{R}` > `twin_unbounded_r{R}`) for each R, and G1 repeated in the
  unbounded variant: P(`twin_unbounded_r{R}` > `twin_unbounded_r1`).
- **Descriptive:** ratio IQM(R) / IQM(R = 1) per family (bounded twin, unbounded twin, hybrid), with its bootstrap
  CI, to compare the relative size of the gain of OR.
- **"Does not learn"** = IQM <= 12 in both metrics (CartPole floor ~9.5; random policy ~22).

## 6. Reading rule (fixed before launching, as written)

1. If the bounded twin learns without OR (R = 1 above 12) and OR gives it no gain (G1 near 0.5) while it does give
   it to the hybrid (G2), the benefit of OR is specific to the features the circuit produces in this set-up.
2. If OR gives the twin a strong gain (G1 strong), OR acts as a generic optimisation mechanism on bounded features
   followed by a linear head, and stops being specific to the circuit. Then compare the relative size of the gain
   (ratio) with that of the hybrid, without claiming more than the data give.
3. G4: if OR helps the bounded twin and does not help (or helps less) the unbounded one, the effect is tied to the
   bounded range of the features, coherent with the explanation of Skolik et al. (2022) that the bounded range of the
   measurement limits the Q-network; if it helps both equally, it does not depend on the range.
4. If the twin does not learn at any R, the contrast is poorly informative and is reported as such.

Always report which branch holds, which do not and why; if none holds cleanly, say so.

### Operationalisation of the quantifiers (our choice, made here, before any cell runs)

The rule above does not say over how many R a condition must hold. `exp12_analysis.py` applies it mechanically with
the following choices (the per-R verdicts of every contrast are always reported in full, so nothing is hidden by the
aggregation; this follows the convention used for the previous control):

- Per-R verdict: *strong*, *strong (reverse)*, *near 0.5*, *indicates a direction (a > b | b > a)* or *mixed*, exactly
  as defined in section 5.
- "The bounded twin learns without OR": IQM `final_perf` > 12 **and** IQM `greedy_final` > 12 at R = 1.
- "G1 near 0.5": near 0.5 at **all four** R in {4, 8, 16, 32}. "G1 strong": strong at **at least one** R.
- "OR gives the hybrid a gain" (G2): strong at **at least one** R.
- Branch 3, evaluated at each R in {4, 8, 16, 32}: *tied to the range* if G1 is strong and (G1 in the unbounded
  variant is not strong, or G4 is strong in favour of the bounded twin); *does not depend on the range* if G1 and G1
  of the unbounded variant are both strong and G4 is near 0.5.
- Branch 4: the bounded twin has IQM <= 12 in both metrics at **every** R in {1, 4, 8, 16, 32}. The unbounded twin
  is reported next to it.
- Anything between (intermediate P values, conditions met only in part) is stated as such, with the direction; the
  analysis does not force a branch.

## 7. Cost

Networks of 132-1124 parameters: about 50 s per cell for the existing classical cells. 100 cells, a few tens of
minutes of wall clock with a handful of workers.

## 8. Outputs

`per_seed_exp12.csv` (same columns as `per_seed_metrics.csv` of the dossier, including the references `hybrid_fig4`
and `hybrid_OR{R}`), `curves_exp12_train_ma50.csv`, `curves_exp12_eval_greedy.csv`, `summary_exp12.csv` (with the
measured parameters), `tables_exp12.md` and `DOSSIER_EXP12.md` (design, structural-equivalence check, parameters,
IQM per arm, tables G1-G4, ratios, application of the reading rule, limitations). Produced by
`experiments/exp12_classical_twin.py` (runner) and `experiments/exp12_analysis.py`; tests in
`tests/test_exp12_arms.py`.

## Amendments

**2026-11-06 - run record (no change to arms, contrasts or reading rule).** The 100 cells (2 families x R in
{1, 4, 8, 16, 32} x seeds 1-10) ran with 5 workers, each with its own pair of seeds and no cooperative locks, in about
11 minutes (about 33 s per cell with five in parallel). No incident: one manifest per cell, no
error in any log, 10 seeds in every arm. The runner was smoke-tested (1 seed, 5 000 steps) and the full test suite
(326 passed, 1 skipped) was run before launching. The analysis applies the operationalisation of section 6 mechanically.
Observation made when reading the result, not a change of rule: the mechanical label of branch 3 at R = 16 ("tied to
the bounded range") rests on the unbounded twin's G1 being 0.92 in `greedy_final` (just under the 0.95 threshold)
while the direct bounded-vs-unbounded contrast (G4) is near 0.5 at every R; the dossier reports both and does not
read the R = 16 label as evidence of range dependence.
