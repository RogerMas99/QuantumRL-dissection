# Experiment 01 - Capacity-matched classical control under DQN

## 1. Where this sits

The reference paper evaluates three blocks of a PQC-based QRL pipeline -
post-PQC inference, observation embedding, ansatz design - under a single fixed
protocol: PPO on CartPole-v1, 100k steps, 10 seeds. The protocol is fixed on
purpose; holding the algorithm constant is what makes component-wise attribution
possible.

That leaves an obvious question the paper does not address: **do the conclusions
depend on the RL algorithm?** The repository ships a functional `dqn.py` that
never produced published results. The author confirmed by correspondence that
DQN was left out for operational reasons - it needed more steps before showing a
useful policy, which combined badly with the cost of training circuits - and not
because the line had been explored and failed.

Extending the dissection off-policy is therefore a genuine open direction. This
experiment is the first step, and it exists because the naive version of that
extension does not work.

## 2. The problem

Porting the paper's five representative configurations to DQN produces a
uniform, architecture-independent collapse:

| config | PPO (n=3) | DQN (n=3) |
|---|---|---|
| OR Classical | 91.1 +/- 9.2 | ~9.7 |
| OR Hsiao | 47.6 +/- 9.2 | ~9.6 |
| Entanglement Hsiao | 22.4 +/- 1.7 | ~9.6 |
| DR Salinas/UQC | 31.3 +/- 4.0 | ~9.4 |
| DR Skolik | 52.5 +/- 7.1 | ~9.7 |

Two readings shaped everything that followed.

**~9.5 is not "failing to learn": it is worse than not learning.** A random
policy on CartPole returns ~22. A return of ~9.5 corresponds to always pushing
in the same direction. The agent is actively learning a degenerate policy.

**The failure is uniform and architecture-independent** - classical and quantum
alike, with and without entanglement, 1 or 5 layers, 1 or 4 qubits, with
near-zero variance across seeds - while PPO separates the same architectures
cleanly. When five architectures fail identically, the cause is upstream of the
architecture, in what all five share: the algorithm's configuration.

That is already a result about **non-identifiability**: in this regime the
dissection cannot attribute anything to any block, because every block reads the
same.

## 3. What has been established so far

Four experiments, all classical unless stated, 3 seeds each.

| configuration | learns? | effect of FIX-01 | informative about FIX-01? |
|---|---|---|---|
| MLP 120-84, batch 16 | yes (~130) | -11 | yes |
| MLP 120-84, batch 128 | yes (~212) | +1 | yes |
| MLP 120-84, 60k steps | yes (~226) | +39 (sd 64-86) | yes |
| paper linear, train_frequency=10 | no (~9.8) | -0.0 | no |
| paper linear, train_frequency=1 | no (~10) | -0.4 | no |
| hybrid Fig. 4 config, 103k steps | weakly (peak 57) | not tested | - |

**The methodological point that drives this experiment:** a delta measured on a
dead agent is uninformative. If learning does not happen for an unrelated
reason, removing poisoned transitions cannot improve something that is not
occurring. Only the top three rows test FIX-01 at all, and there it is zero
within noise.

So the honest status is: the phantom-transition mechanism is verified and
present in the published environment, but its causal role in the ~9.5 collapse
is **not currently supported by any experiment run inside the real framework**.
The only evidence for that role comes from a numpy reimplementation that differs
from the framework in at least three ways (Double DQN, gradient clipping, buffer
size).

## 4. Why the arms cannot simply be given more capacity

The paper's classical control (section 3.4A) is the hybrid arm minus the PQC:
matched input dimensionality, matched parameter count, identical linear readout.
Section 3.3 is explicit that no hidden layers are used in the default inference
block. The design exists so that the only difference between arms is the
circuit.

That constraint is not incidental, it is the paper's thesis. Put an
11k-parameter MLP behind the circuit and the MLP solves CartPole by itself: the
five configurations would return near-identical curves and the dissection would
measure nothing. The paper's own Output Reuse finding - that OR helps hybrid
agents and not classical ones - only holds *because the head is small*.

The constraint runs through the whole design, not just the classical arm: the
hybrid configuration of Fig. 4 is `net_arch=[4]` with `activation=Identity`,
which composes to a single linear map.

**But the constraint carries an implicit requirement: that both arms are alive.**
Under PPO they are. Under DQN the classical arm dies, and comparing a hybrid arm
at 57 against a dead classical arm says nothing about the PQC - the death may
come from DQN interacting with a linear Q-function (deadly triad, bootstrapping
to ~100 from a couple dozen parameters) rather than from the absence of a
circuit. The paper's parity argument is valid on-policy and breaks off-policy.

## 5. The design

The capacity ladder, measured rather than assumed:

| arm | features -> Q | trainable params | observed |
|---|---|---|---|
| paper linear | 12 (3x4 reuse) -> linear | 26 | ~10, dies |
| matched_classical | 12 -> 8 -> linear | 122 | to measure |
| hybrid Fig. 4 | PQC (8q x 5L, 80 w) -> linear head (46) | 126 | 57, learns weakly |
| oversized MLP | 4 -> 120 -> 84 -> linear | 10,934 | 226, healthy |

Parameter counts above are measured (capacity ladder), not estimated. The
hybrid's 126 splits as 80 quantum weights + 46 classical head. `matched_classical`
is sized to the hybrid TOTAL (126), giving hidden width 8 and 122 parameters -
so any gap to the hybrid is attributable to the circuit, not to a parameter
advantage. The hybrid circuit itself is left exactly as in the paper, since it is
the object of study and must stay comparable to the PPO runs.

The gap between arm 1 and arm 2 conflates two things: the presence of a circuit,
and the amount of trainable machinery. **NEW-02 separates them** by adding an arm
that keeps the paper's design - same reduced observation, same linear readout -
but replaces the PQC with a classical block of *the same measured parameter
count*.

Four arms, each answering a different question:

| arm | question it answers |
|---|---|
| `paper_linear` | reproducibility audit: what happens to their exact design under DQN? |
| `matched_classical` | the fair comparison: at equal parameter budget, does the circuit contribute? |
| `hybrid_fig4` | the object of study |
| `oversized_mlp` | labelled control only: is the environment/algorithm setup sound at all? |

`oversized_mlp` must never be compared to the hybrid arm - 90x the parameter
budget. Its only job is to close the objection that the DQN implementation is
simply broken.

The grid crosses every arm with FIX-01 on and off, 3 seeds, 60k steps.

## 6. How to read the outcomes

**`matched_classical` learns.** The death of `paper_linear` was capacity, not the
absence of a circuit. Two consequences: the paper's parity design is not viable
off-policy and needs re-specification, and we finally gain a second *live*
configuration in which FIX-01 becomes measurable.

**`matched_classical` dies like `paper_linear`.** At equal parameter budget the
classical block does not reach where the PQC does. This is a positive result for
the circuit and the strongest available claim about it under DQN - stronger than
anything the paper reports, since the paper never establishes that the circuit
is necessary rather than merely sufficient.

**Mixed across seeds.** Turn it into a variance result. Seed dispersion in this
regime is already known to be large: the same MLP cell at batch 128 returned
141, 213 and 284 across three seeds, a factor of two from seed alone. This
matters independently, because the published Fig. 4 is a single seed with no
variance band.

All three are publishable. None depends on attributing anything to FIX-01.

## 7. Metric convention

Report `best_ma50`, the best 50-episode moving average, not the tail mean. DQN
on CartPole decays from its peak even when healthy - deadly triad, epsilon
pinned at `end_e` - so a last-N-episodes statistic reports failure on runs that
reached 200-400 mid-training. Where the greedy evaluation hook (NEW-03) is
enabled, `greedy_best` is primary.

Baselines: random policy ~22; degenerate constant-action policy ~9.5. The
distinction matters, since a run at ~9.5 is a different diagnosis from a run
at ~22.

## 8. Threats to validity

- **n = 3.** Adequate to separate ~10 from ~200; inadequate for anything
  finer. Seed dispersion at the top of the range is a factor of two.
- **One environment.** CartPole-v1 only, as in the paper. Nothing here
  generalises past it without further work.
- **FIX-02 is untested in effect.** It is verified to reach the model, but
  whether it changes hybrid outcomes is an open question and a live suspect for
  the hybrid arm's ceiling.
- **The upstream history before 2025-06-19 is private.** Claims about what the
  code has always done are limited to the public record.
- **Capacity matching is by parameter count**, which is one notion of parity
  among several. Matching by wall-clock, by number of gradient updates, or by
  expressibility would give different arms. Parameter count is what the paper
  itself uses in section 3.4A, which is why it was chosen.

## 9. Before running the full grid

1. `python scripts/verify_env.py` - FIX-04 checks.
2. `pytest -q` - probe, capacity and compat tests.
3. `--ladder-only` - confirm `paper_linear` really builds `Linear(12, 2)` with
   ~26 parameters. If the count is wrong, the arm is not what we think it is and
   the twelve runs would be worthless.
4. One short run (2k steps) to measure throughput before committing hours.

## 10. Open items

- Nothing in this repository has been executed end to end on the target
  hardware. The classical grid is cheap; run it first.
- Reproduce the working numpy configuration (buffer 50k, train_frequency 1,
  batch 16, Double DQN, gradient clipping) inside the framework, and diff that
  harness against `dqn.py` to identify which lever actually rescues the linear
  arm.
- Hybrid arms are deferred until at least one classical arm both learns and
  shows a measurable FIX-01 effect. Until then, hybrid runs cost hours to
  produce uninterpretable deltas.
