# Experiment 09 — Output Reuse x depth on FrozenLake, on the SU(2) emulator

> **STATUS: EXPLORATORY EXTENSION — NOT PART OF THE THESIS CORE.**
> This experiment explores extra options that the core design (exp01–exp08,
> and the controls around them) did not cover. It is run because it is cheap,
> not because the thesis depends on it. Nothing here is a pre-registered test
> of the core hypotheses; results are to be reported separately and labelled
> exploratory, unless the supervisor decides to promote any of it into the
> core. Created 2026-10-20.
>
> **Everything in this experiment comes from simulators (`su2`, `sv`). Under
> [SIMULATED-RESULTS-POLICY.md](SIMULATED-RESULTS-POLICY.md) none of it can go into
> the final work, in any form, without the author's explicit approval.**

**Axes:** algorithm `dqn` x environment `frozenlake` x blocks 1 (Output Reuse) and 2 (DR / embedding)
**Script:** `experiments/exp09_dqn_emulator_sweep.py`
**Arms:** `su2_frozen_{scalar_1q,binary_4q}_L{L}[_OR{R}]` (24, `core/configs.py`, section "[exp09]")
**Results:** `results/exp09_dqn_emulator_sweep/` (Drive)

---

## 0. Why this exists

- On CartPole, Output Reuse is the strongest single lever in our data
  (`hybrid_OR{4,..,32}` against `hybrid_fig4`: P between 0.96 and 1.00), and the only
  setting in which entanglement showed an advantage over its unentangled twin
  (provisional: the unentangled OR arms are still at n = 5–9).
- It has never been tried on FrozenLake. That is a matter of timing, not of a
  decision against it: exp04 was designed while exp02 (OR) was still running and
  exp03 (DR) was the block that had transferred (`docs/EXPERIMENT-04.md`, section 1).
- `OutputReuse` is `x.repeat(1, R)` (`simplyqrl/agents.py`): R identical copies
  of the measured expectation values. It does not enlarge the function class of
  the head; it re-parametrises it. Whatever it does on FrozenLake is therefore
  an optimisation effect, a hypothesis this experiment can inform but not settle.

## 1. What the emulator can and cannot say

`SU2SkolikEmulator` reproduces the UNENTANGLED skolik circuit exactly (NEW-05;
initialisation matched in FIX-12) at 3–8 minutes per 100k-step FrozenLake cell,
against 7–25 hours for the real circuit. It cannot represent an entangled
circuit. **This sweep says nothing about entanglement.** The crossing Output
Reuse x entanglement needs the real circuit (about 25 h per seed on Config B,
L = 5) or a purpose-built batched statevector simulator: phase 2, gated on what
this phase shows.

## 2. Design (wave 1)

| group | what | arms | seeds | cells | est. compute |
|---|---|---|---|---|---|
| **F** | Config A (1 qubit, scalar embedding): L in {1, 5, 10, 15} x R in {1, 4, 16} | 12 | 1–10 | 120 | ~14 h |
| **G** | Config B (4 qubits, binary embedding, ent=False): L in {1, 2, 3, 5} x R in {1, 4, 16} | 12 | 1–10 | 120 | ~8 h |
| **I** | protocol sensitivity: train_frequency {1, 10} x learning_starts {1k, 10k} x buffer {50k, 10k} on `su2_frozen_scalar_1q_L5` and `su2_frozen_binary_4q_L5` | 2 x 8 variants | 101–103 | 48 | ~5 h |

- R = 1 means no Output Reuse. At L = 1 the entangled and unentangled circuits are
  the same circuit (FIX-07), so Config B L = 1 is fully covered.
- **Protocol.** F and G use exp04's own FrozenLake protocol (batch 128, buffer
  50 000, train_frequency 1, learning_starts 1 000) so that they stay comparable
  with every FrozenLake result already on disk. I is a **sensitivity analysis**
  run on separate seeds (101–103); it is not used to choose the protocol of F
  and G. If it shows strong sensitivity (as the L = 5 Fourier ceiling did, FIX-13),
  that is reported as a caveat and, if worthwhile, F/G are repeated under the
  better protocol as a clearly labelled secondary set.
- 100 000 steps, FIX-01 on, greedy evaluation every 10 000 steps (20 episodes).
- Cells are run by several workers of one group with `--claim` (cooperative
  locks, TTL 2 h).

## 3. Contrasts fixed before any cell runs

Primary outcome `greedy_final`; secondary `final_perf`; probability of
improvement P(a > b) with bootstrap CIs on the IQM, as in the rest of the repo.
Six contrasts are treated as the confirmatory set of this exploration; every
other comparison is reported as exploratory.

1. Config A, L = 5: R = 16 against R = 1.
2. Config A, L = 10 and L = 15: R = 16 against R = 1.
3. Config B (ent=False), L = 5: R = 16 against R = 1.
4. Config B (ent=False), L = 1: R = 16 against R = 1.
5. Dose–response: R in {1, 4, 16} at each (config, L) — monotone or not.
6. Emulated Config A circuit against the Fourier ceiling (`__flkw`, FIX-13) at
   matched L, to see whether OR closes any of the ceiling gap.

Reading rules: an effect on a configuration that is dead at R = 1 and alive at
R > 1 is the interesting outcome; no effect anywhere is a result and is reported
as such. A gain on `final_perf` with no gain on `greedy_final` is the value-vs-
policy dissociation (greedy evaluation on deterministic FrozenLake has one
effective sample per checkpoint) and is not read as learning.

## 4. Not in wave 1

- **Control: a classical MLP on the four bits with a matched budget.** Missing
  from the FrozenLake design since exp04; it needs a new observation adapter
  (state as bits) and is not part of this wave.
- **CartPole mechanism of OR** (head size, learning rate, a classical control that
  repeats its inputs), new DR x OR points and per-depth Fourier ceilings: a second
  wave, only if wave 1 or the supervisor makes them worth the time.
- **Output Reuse x entanglement on the real circuit** (phase 2).

## 5. Results (wave 1, 288 cells, finished 2026-10-21)

All 288 cells ran (groups F and G at n = 10, group I at n = 3 per variant), no errors.
Exploratory, as stated at the top: nothing here changes the core of the thesis.

**Headline: Output Reuse rescues nothing on FrozenLake, on the unentangled circuit.**

- **None of the 24 arms of F and G is alive.** `greedy_final` is 0.000 and 0 of 10 seeds
  are alive in every one of the 240 cells, for every depth and every R in {1, 4, 16}
  (Config A: L = 1, 5, 10, 15; Config B ent=False: L = 1, 2, 3, 5).
- The only movement is in `final_perf` of Config A (largest IQM 0.102, at L = 5, R = 1),
  and it goes the wrong way for OR: 0.102 / 0.026 / 0.048 for R = 1 / 4 / 16 at L = 5,
  0.046 / 0.029 / 0.007 at L = 10. With `greedy_final` at zero this is the value-vs-
  policy dissociation already seen in FrozenLake, not learning.

### Pre-specified contrasts (section 3)

| # | contrast | greedy_final a / b | P(a>b) greedy | final_perf a / b | P(a>b) final |
|---|---|---|---|---|---|
| 1 | Config A L5: R16 vs R1 | 0.000 / 0.000 | 0.50 | 0.048 / 0.102 | 0.40 |
| 2 | Config A L10: R16 vs R1 | 0.000 / 0.000 | 0.50 | 0.007 / 0.046 | 0.22 |
| 2 | Config A L15: R16 vs R1 | 0.000 / 0.000 | 0.50 | 0.014 / 0.010 | 0.58 |
| 3 | Config B L5 (ent=False): R16 vs R1 | 0.000 / 0.000 | 0.50 | 0.000 / 0.000 | 0.45 |
| 4 | Config B L1: R16 vs R1 | 0.000 / 0.000 | 0.50 | 0.000 / 0.000 | 0.45 |
| 5 | dose-response R = 1, 4, 16 | no arm moves on `greedy_final` | — | no consistent direction | — |
| 6 | Fourier ceiling (`flkw`) vs emulated circuit, L = 10 and 15, every R | 1.000 vs 0.000 | 1.00 | 0.949 / 0.950 vs <= 0.046 | 1.00 |

P = 0.50 on `greedy_final` is two groups that are both at zero: it says no effect was
found, not that the arms are equivalent in any finer sense.

### Contrast 6 in words
The gap between the classical additive Fourier ceiling (10 of 10 seeds alive at L = 10 and
15) and the circuit is **not closed by Output Reuse at any R**. At L = 5 both are dead
under the FrozenLake protocol, so there is no gap to close.

### Validation: the emulator against the real circuit on FrozenLake
Same protocol, seeds 1–10, five pairs (Config A L = 5, 10, 15; Config B L = 1, 5): the
emulated arm and its real PennyLane counterpart are indistinguishable (P(emulator >
real) between 0.45 and 0.56 on both metrics). Caveat: all of these arms are dead, so the
check has little power on FrozenLake; the stronger validation is the CartPole one
(FIX-12, P = 0.41 / 0.42 for an arm that is alive).

### Group I: protocol sensitivity (seeds 101–103, n = 3 per variant)
None of the 8 protocol variants brings either L = 5 arm to life: 0 of 3 seeds alive and
`greedy_final` 0.000 in all 16 (arm, variant) cells. The best `final_perf` is 0.148
(`su2_frozen_scalar_1q_L5`, train_frequency 1, learning_starts 10k, buffer 50k). With three
seeds this cannot detect a modest effect, but there is no sign that the failure of the
unentangled circuit at L = 5 is a matter of these three DQN settings.

### Reading
1. On CartPole, Output Reuse helped the unentangled circuit as well as the entangled one.
   On FrozenLake it does not help the unentangled circuit at all. Whatever OR does
   (an optimisation effect, section 0), it did not make an otherwise dead circuit learn here.
2. The only FrozenLake arm that learns is the ENTANGLED circuit at L = 5 (9 of 10 seeds,
   exp04). Because every unentangled baseline is dead and stays dead, this experiment
   cannot show whether OR interacts with entanglement; that is the question phase 2 would
   have to ask, on the entangled arm itself.
3. The emulator does what it is for: it reproduces the real unentangled circuit and made
   240 + 48 cells affordable (about 4.7 hours wall-clock on up to 13 single-thread processes).

### What this does not show
- Anything about entanglement (the emulator cannot represent it).
- That OR is useless in general: only that it did not rescue these 24 unentangled
  FrozenLake arms, at R in {1, 4, 16}, under exp04's protocol.
- Anything about the control "classical MLP on the four bits", still missing (section 4).

### Phase 2 (not launched; decision for the supervisor)
The informative test is Output Reuse on the real entangled circuit: Config B L = 5 with
OR (the only arm that is alive at baseline) and, if budget allows, entangled L = 2–3 with
OR to see whether it lowers the depth needed. Cost is the real circuit's: about 25 h per
seed at L = 5. A batched statevector simulator for the entangled circuit would make it
cheap; it does not exist yet.

---

## 6. Wave 2 (designed 2026-10-21, before any wave-2 cell ran)

Same status as the rest of this document: **exploratory extension, outside the thesis
core, and nothing here replaces a run planned for the core** (the Colab top-ups of exp07
and exp08 stay exactly as planned). Wave 2 uses the batched statevector simulator
(`core/statevector.py`, NEW-09), which represents ENTANGLED circuits, and the SU(2)
emulator for extra axes.

| group | what | cells | est. compute |
|---|---|---|---|
| **P** | Config B ENTANGLED (sv): L in {1, 2, 3, 5} x R in {1, 4, 16}, seeds 1-10. Phase 2: Output Reuse x entanglement on FrozenLake | 120 | ~20 h |
| **V** | CartPole fidelity: `HYBRID_FIG4` (entangled, sv) at R in {1, 4, 8, 16, 32}, seeds 1-10 | 50 | ~10 h |
| **K** | CartPole extras (su2): OR without entanglement R in {4, 8, 16, 32}; output scale (FIX-02) at R = 1, 16; learning rate {1e-3, 4e-3} at R = 1, 16 | 100 | ~3 h |
| **H** | FrozenLake axes never varied, one at a time on the dead arms (Config A L5, L10; Config B L5 unentangled): head16, init0.1, init0.5, trainable input scale, lr {1e-3, 4e-3}; seeds 1-5 | 90 | ~9 h |
| **D** | expressivity and gradient diagnostics without RL (`experiments/exp09b_expressivity_diagnostics.py`) | - | <1 h |

### Fixed before any wave-2 cell runs

1. **Fidelity of the simulator (V and P at L = 5, R = 1).** `sv_cartpole_L5{,_OR*}` against
   the real `hybrid_fig4` / `hybrid_OR*` of exp01 / exp02, and `sv_frozen_binary_4q_L5`
   against the real `frozen_binary_4q_L5` of exp04 (9 of 10 seeds alive). The simulator is
   accepted as a stand-in only if P(sv > real) lies in [0.25, 0.75] on `final_perf` and
   `greedy_final` for each pair, and on FrozenLake at least 7 of 10 seeds are alive. If it
   fails, P and V are reported as a failed validation and nothing else is read from them.
2. **P, the question.** Output Reuse on the entangled circuit: R = 16 against R = 1 at
   L = 5, 3 and 2 (does OR help or hurt the arm that is alive, and does it lower the
   depth needed?); dose-response R = 1, 4, 16; and L = 1 as control (entangled == unentangled).
3. **V and K, the CartPole replication of "entanglement only shows with OR".**
   Entangled (sv) against unentangled (su2) at each R in {1, 4, 8, 16, 32}, n = 10 each,
   same protocol - the contrast that is still at n = 5-9 in the real unentangled arms.
4. **K, the two single-factor checks.** Output scale on against off at R = 1 and 16;
   lr 1e-3 and 4e-3 against 2.5e-4 at R = 1 and 16.
5. **H.** Each variant against the group-F baseline of the same arm on the same seeds
   (paired). The outcome of interest is whether ANY dead arm comes alive.

Reading rules are those of section 3. Diagnostics (D) are descriptive: a circuit that fits
the optimal action values Q* but fails under DQN points to optimisation, not expressivity.

### Not in wave 2
The classical MLP on the four bits (needs an observation adapter), extending the emulators
to the Salinas and Hsiao templates, and any substitution of Colab runs planned for the core.

---

## 7. Validation battery of the simulators (designed 2026-10-21, before any battery cell ran)

> SIMULATOR RESULTS - EXPLORATORY - not for the final work without the author's explicit
> approval ([SIMULATED-RESULTS-POLICY.md](SIMULATED-RESULTS-POLICY.md)).

**Why.** Section 6's check (the per-pair criterion) found 6 of 7 pairs inside the window and
failed one (FrozenLake L = 5, `final_perf`, P = 0.21). One pair is too little to say whether the
simulators reproduce the real circuit. This battery replicates every real arm we already have that
a simulator can represent, and judges the simulators on all of them together. Real runs are not
repeated (25 h per FrozenLake seed): the simulator side is run at n = 10 against the real runs
that exist.

### Pairs (simulator arm vs the real arm it replicates; real seeds 3–10 depending on the arm)

| simulator arm | real arm | source of the real runs | what it tests |
|---|---|---|---|
| `sv_cartpole_L5{,_OR4..32}` | `hybrid_fig4`, `hybrid_OR4..32` | exp01, exp02 | done in section 6 |
| `sv_cartpole_L{1,2}` | `hybrid_DR1`, `hybrid_DR2` | exp03 | depth, entangled |
| `sv_cartpole_L{1,2}_OR{4,8,16,32}` | `hybrid_DR{1,2}_OR{R}` | exp08 | depth x Output Reuse |
| `su2_cartpole_L{1,2}` and `su2_cartpole_L5__init2pi` | `hybrid_DR{1,2,5}_noent` | exp03b | unentangled emulator, depth |
| `su2_cartpole_L5_OR{4,8,16,32}` | `hybrid_or_r{R}_noent` | exp08 | unentangled emulator, OR |
| `sv_paper_skolik_{4q,8q}_L{1,2,5}` | `paper_skolik_{4q,8q}_L{L}` | exp07 | other qubit counts, other embedding, linear head |
| `su2_paper_skolik_4q_L{2,5}_noent` | `paper_skolik_4q_L{2,5}_noent` | exp07 | same, unentangled |
| `su2_frozen_scalar_1q_L{5,10,15}`, `su2_frozen_binary_4q_L{1,5}` | `frozen_scalar_1q_L*`, `frozen_binary_4q_noent_L*` | exp04 | FrozenLake, unentangled (already run in wave 1) |
| `sv_frozen_binary_4q_L{1,5}` | `frozen_binary_4q_L{1,5}` | exp04 | FrozenLake, entangled (section 6, plus seeds 11–20) |
| `sv_cartpole_L5_noent`, `sv_frozen_binary_4q_noent_L{1,5}` | `su2_*` twins | this experiment | the two simulators against each other |

### Noise floor
Two REAL arms that are the same configuration run in different experiments give the size of
disagreement that real-vs-real already shows: `hybrid_fig4` (exp01) against `hybrid_DR5` (exp03)
(`HYBRID_FIG4` with `n_layers_q = 5` is the same circuit). The simulators are judged against that
reference, not against zero.

### Metrics per pair
`final_perf`, `greedy_final` and the area under the 50-episode moving-average curve (`auc_ma`,
which sees the whole learning curve, not only its end); P(sim > real), a permutation p-value, and
the standardised difference d = (mean_sim - mean_real) / pooled sd on `final_perf`.

### Acceptance rule, fixed before the battery runs
The simulators are accepted as stand-ins for the real circuit **only if all three hold**:
1. **No global bias.** The mean standardised difference d over all INFORMATIVE pairs has a 95 %
   bootstrap CI that contains 0, and |mean d| <= 0.30.
2. **No excess of disagreements.** At most 25 % of the (informative pair, metric) cells have
   P(sim > real) outside [0.25, 0.75]. (With 10 against 10 seeds about 7 % are expected outside it
   by chance.)

*Informative pair* (defined before any battery cell finished): at least one of the two arms has an
IQM of `final_perf` above 12 on CartPole or above 0.05 on FrozenLake. Otherwise both arms sit at the
dead floor, where a standardised difference or a probability of improvement measures noise; such
pairs are reported, and the only check applied to them is "both dead". Pairs that reuse the same
simulator arm against a second real reference, and the real-vs-real noise floor, are reported but
do not enter the global rule, so that no simulator data is counted twice.
3. **The two simulators agree with each other.** For the `sv` ent=False / `su2` twins,
   P in [0.25, 0.75] on every metric.

The per-pair verdicts of section 6 are kept as they are (one FAIL). A failure of this rule means
the simulators are not accepted and nothing read from them is interpreted. A pass means they are
accepted as exploratory tools, still under the policy above: it does not make their results
usable in the final work.

### Result of the battery (2026-10-21)

Full table: [EXPERIMENT-09-BATTERY.md](EXPERIMENT-09-BATTERY.md). 21 informative pairs, 63
(pair, metric) cells, 9 twin cells.

| criterion | outcome | verdict |
|---|---|---|
| 1. no global bias | mean d = +0.119, 95 % CI (+0.010, +0.250) | **FAIL** (CI excludes 0 by 0.010) |
| 2. at most 25 % of cells outside P in [0.25, 0.75] | 3 of 63 = 5 % | PASS |
| 3. the two simulators agree | 0 of 9 twin cells outside the window | PASS |

**Not accepted under the rule as written.** The simulators are, however, close: the mean
standardised difference is +0.12 (the rule's bound on its size is 0.30), 95 % of the
cells are inside the window against ~93 % expected by chance, the real-vs-real noise floor (two real
runs of the same configuration) is d = -0.11 with P = 0.49 / 0.34 / 0.48, and the SU(2) and
statevector routes agree with each other. Where the disagreement concentrates: entangled CartPole
with Output Reuse at R = 16 and 32 (simulator higher, d about +0.5 each), `sv_paper_skolik_8q_L2`
(+1.05, real n = 4), and FrozenLake L = 5 (simulator lower: d = -0.10 on 10 seeds, -0.45 on all
20). The direction differs between environments, which a single systematic bug would not do.

The rule was not relaxed after seeing this. Whether to accept the simulators for a narrower use
(for example "effects larger than half a standard deviation") is the author's decision, to be
recorded in [SIMULATED-RESULTS-POLICY.md](SIMULATED-RESULTS-POLICY.md) if taken.

