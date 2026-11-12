# Experiment 04 — Data Reuploading and observation embedding under DQN (FrozenLake-v1)

**Axes:** algorithm `dqn` × environment `frozenlake` × block 2 (embedding / DR)
**Script:** `experiments/exp04_dqn_frozenlake_embeddings.py`
**Notebooks:** `notebooks/00_fix05_verification.ipynb` (gate), `notebooks/04_dqn_frozenlake_runner.ipynb`
**Status:** stage 0 measured; stage 1 done at n=10 (liveness gate PASSES,
`frozen_onehot_mlp` 0.999 — see the 2026-08-21 update to section 5); stage 2
underway (`frozen_scalar_1q_L1` at 8/10 seeds, the rest of the grid launching
now, "exp04b")

---

## 0. Two papers, and which one says what

This distinction governs every claim below, so it goes first.

| | **Dissection paper** (arXiv:2511.17112) | **SimplyQRL chapter** (Springer, 2025) |
|---|---|---|
| what it is | systematic evaluation of three pipeline blocks | library announcement with illustrative demos |
| algorithm | PPO only | PPO (Exp 1–3), one hybrid DQN curve (Exp 1, Fig. 4) |
| environment | **CartPole-v1 only** | CartPole (Exp 1–2), **FrozenLake (Exp 3)** |
| protocol | 100k steps, 10 seeds, controls | Exp 3: 500k steps, **single seed**, no controls |
| blocks dissected | OR, DR, ansatz | none — the experiments demonstrate configurability |

**The dissection paper never touches FrozenLake.** The FrozenLake material lives
in the library chapter, where it is an illustrative demonstration that the
configuration dictionary works: two built-in transforms, a DR-depth sweep, one
seed, no classical control, no block-level conclusion drawn.

This corrects the framing an earlier draft of this document used. The standing
question of this thesis — *do the dissection paper's block-level conclusions
survive a change of algorithm?* — **cannot be asked of FrozenLake**, because
there is no dissected FrozenLake result to transfer from. Fig. 6 of the chapter
is a sanity anchor, nothing more, and a single-seed one at that.

## 1. So what is this experiment actually for

Three things, none of which needs a FrozenLake dissection baseline.

**(a) Generality of exp03, the one block that transferred.** exp01 returned a
negative under DQN (no clear circuit advantage at equal budget and information)
and exp02 is still running, but **exp03 found strong monotone DR transfer**:
greedy 15 → 35 → 199 for depths L = 1/2/5 at 100k steps on CartPole. That is the
thesis's one clean positive so far, and it rests on a single environment. A DR
sweep on a second, structurally different environment — sparse reward, discrete
state, short episodes — is what turns "DR transfers to DQN" into "DR transfers to
DQN, and not only on CartPole". If it fails to replicate, that is equally worth
knowing and bounds the exp03 claim rather than inflating it.

**(b) FIX-01 in a regime where it is measurable.** The strongest reason, and it
depends on no paper at all. The phantom fraction FIX-01 removes is approximately
`1 / mean_episode_length`, so on CartPole it *shrinks as the agent improves* —
under 1% once an arm learns well, which is exactly why exp01 found no significant
FIX-01 effect in any live arm despite verifying the mechanism. FrozenLake inverts
this: the optimal path is six steps and episodes end in a hole or at the goal, so
mean length stays near 6–10 for the entire run and the phantom fraction stays
near 10–15% regardless of performance.

The poison is also better aimed. The phantom transition is
`(terminal_state, arbitrary_action, r=0, s'=start, done=False)`. Under dense
reward it is one corrupted sample among many informative ones. On FrozenLake the
reward is a single bit delivered only at the goal, and the phantom attaches to
precisely the transition that carries it — teaching the network that reaching the
goal is followed by bootstrapping from the start with zero reward.

A large FIX-01 effect here against the CartPole null is a **mechanism result**:
the bug's impact scales with episode turnover, predicted quantitatively before
measurement. That converts exp01's negative into a positive with a story.

**(c) A properly controlled version of the chapter's Exp 3.** Config A and
Config B were run once each, single seed, no classical control. Running them with
matched-budget controls and 8–10 seeds is a contribution in itself, and it is
cheap: Config A is a **one-qubit** circuit and Config B four, against CartPole's
eight. The robustness pass plan B has deferred on exp01/02/03 is affordable here
for the first time.

## 2. Open decision: does this trigger the modernisation commitment?

`docs/ROADMAP.md` carries a standing commitment: moving from reproducing the
paper to a genuinely new setting is the moment to branch, update the stack and
re-verify the baseline from scratch. Its trigger was described as "a new method,
or taking it to Acrobot/LunarLander".

FrozenLake sits between the two cases and an honest reading has to say so. It is a
new environment for the *dissection*, so on the letter of the rule it counts; but
it is an environment the library itself ships and demonstrates, so there is a
published curve to sanity-check against.

**Recommendation: keep the pinned stack — for a reason that has nothing to do
with which paper FrozenLake appears in.** Point (b), the headline, is a
*cross-experiment* comparison: FIX-01's effect here against exp01's null on
CartPole. That comparison is valid only if both sides share the stack. Changing
gymnasium between them would make the central claim unfalsifiable, and upgrading
would not help anyway, since the autoreset behaviour is an intentional gymnasium
design decision that newer versions entrench.

Acrobot has no such tether and remains the right place for the branch.

*This is a judgement call and should be confirmed explicitly rather than inherited
from this document.*

## 3. Blocker: FIX-05

Upstream SimplyQRL cannot run FrozenLake under DQN at all. Three symptoms, two of
them silent. See `docs/CORRECTIONS.md#fix-05`, and run
`notebooks/00_fix05_verification.ipynb` — which reproduces the bug from scratch,
verifies the fix and prints the capacity ladder — before anything else.

The fix is an environment adapter, not a patch to upstream internals:
`Box(shape=(1,), float32)`, state index unchanged in value, reproducing PPO's
effective representation exactly.

## 4. Design

Environment: `FrozenLake-v1`, `map_name="4x4"`, `is_slippery=False`,
`max_episode_steps=100`, through `DiscreteToBoxObs` (scalar arms) or `OneHotObs`
(classical reference arm).

### Arms

| arm | type | observation | params | role |
|---|---|---|---|---|
| `frozen_scalar_1q_L{1,5,10,15}` | hybrid, 1 qubit, `ent=False` | scalar → phase | 10 / 18 / 28 / 38 | chapter Config A |
| `frozen_binary_4q_L{1,5}` | hybrid, 4 qubits, `ent=True` | binary basis | 28 / 60 | chapter Config B |
| `frozen_binary_4q_noent_L{1,5}` | hybrid, 4 qubits, `ent=False` | binary basis | 28 / 60 | ablation, ours |
| `frozen_matched_scalar` | classic | scalar | 22 | **fair control**: matched budget and information |
| `frozen_scalar_mlp_large` | classic, oversized | scalar | 4548 | separates encoding from capacity |
| `frozen_onehot_mlp` | classic, oversized | one-hot (16) | 5508 | **liveness guard** |

All arms are registered in `core/configs.py :: ARMS`, per the repo's own
placement rule, so exp04 uses `RunSpec` / `run_grid` exactly like exp01 - **no
changes to `SafeDQN`, `runner.py`, `capacity.py`, `analysis.py` or `compat.py`**.
The matched control is computed from the reference hybrid's measured budget, then
rounded *up*: `match_hidden_width` returns the closest width, which at these tiny
budgets lands under (18 → width 2 → 16 params). A control that loses while
carrying fewer parameters proves nothing, so the conservative direction wins.

A structural note that belongs in the write-up: Config A's head is
`Linear(1, 4)`, so all four Q-values are affine functions of a single expectation
value. That is a limit of the chapter's configuration, not of this set-up.

`ent=True` is invalid at one qubit — the circular CZ becomes `CZ(wires=[0, 0])`.
Config A is therefore *necessarily* unentangled. This matters: comparing Config A
against Config B confounds embedding with entanglement, and the chapter does not
flag it. The `noent` ablation separates them, and it is one flag away.

### Stage 0, measured

Run during integration on the pinned stack, and already in `RESULTS-LOG.md`:

| quantity | value | CartPole reference |
|---|---|---|
| random-policy success rate | 0.0150 | — |
| mean episode length (random) | 7.69 | ~22 at random, 200-500 once learning |
| predicted phantom fraction `1/len` | 0.130 | < 0.01 at convergence |
| **measured** phantom, one-hot arm, 6k steps | 0.086 off / 0.099 on | < 0.01 |

The premise holds: the phantom fraction here is an order of magnitude above
CartPole's, and unlike CartPole it does not shrink as the agent improves.

Throughput, same runtime: ~1600 steps/s classical, ~50 at one qubit, ~25 at four.

### Hyper-parameters

exp01 defaults except where FrozenLake forces a change. Each deviation recorded,
because an unexplained hyper-parameter difference between environments is a
confound a reviewer will find.

| parameter | value | why it differs |
|---|---|---|
| `train_frequency` | 1 | exp01: the upstream default of 10 gradient-starves small networks enough to mask every other effect |
| `buffer_size` | 50 000 | with one bit of reward, a buffer that forgets rare successes starves the only signal there is |
| `learning_starts` | 1 000 | 10 000 is a large fraction of a run at these episode lengths |
| `batch_size`, `gamma`, `lr`, `target_network_frequency` | 128, 0.99, 2.5e-4, 500 | unchanged |

### Metric

FrozenLake's episodic return is a single bit, so `best_ma50` is a noisy estimator
here. The primary training metric is the **success rate**: a 100-episode rolling
mean of `ep_reward`, directly interpretable as the fraction of recent episodes
reaching the goal. `greedy_best` stays the primary reported metric, as in exp01.

**UPDATE 2026-08-30 - a mechanism for why this matters more here than in
exp01: a separation between the LEARNED VALUE and the INDUCED POLICY, not
"the agent learned nothing."** `docs/RESULTS-LOG.md`'s exp04 stage-2 update
diagnoses directly (retrained a cheap arm fresh, with weights now genuinely
saved and inspected, not inferred from aggregate numbers) why `greedy IQM`
sits at 0.000 across most stage-2 arms despite real, non-spike training
success (`final_performance` up to 0.199-0.305): `FrozenLake-v1` with
`is_slippery=False` is fully deterministic, so a deterministic (epsilon=0)
policy either reaches the goal, falls in a hole, or enters an absorbing CYCLE
and burns the full `max_episode_steps=100` doing neither. The Q-network's
VALUES need not be degenerate for this to happen - `frozen_scalar_mlp_large`
carries large, structured Q's (~18-19) across every state - it is the
**argmax**, the induced policy, that falls, by a razor-thin margin (measured:
0.165 out of ~18.7 at the trapping state, under 1%), on the side of a
zero-reward absorbing loop rather than any path to the goal. See
`docs/RESULTS-LOG.md`'s exp04 stage-2 update for the exact Q-value table and
the verified greedy rollout.

**Three runs, two trajectories, not one repeated description.** Three
from-scratch re-runs of the identical spec (same arm, seed, `DQN_KWARGS`,
`total_timesteps`) produced: the original session's now-unreproducible
description (`0<->4`, DOWN/UP, "~0.01-0.02"), a second run landing in a
DIFFERENT trap (`0->1->2->3` then a self-loop at 3, RIGHT against the grid
edge, gap 0.165), and a third run landing back on `0<->4` with the gap this
time actually measured (0.002 at state 0, 0.018 at state 4 - see
`docs/RESULTS-LOG.md`'s exp04 stage-2 update for the full table). That is
`docs/CORRECTIONS.md#fix-10`: `DQN(seed=...)` never seeds epsilon-greedy's
action sampler, so this project's reproducibility is statistical (N seeds, an
IQM, a CI), not bit-identical - re-running one nominal seed is not guaranteed
to replay the same trajectory.

**Read this as a count, not a rate.** `0<->4` has now recurred (twice) where
the state-3 self-loop has not (once), but three runs from one machine are
not equivalent samples of one population - nothing here records what else
may have differed between them, and FIX-10 is precisely why the nominal
seed cannot be assumed to equalise that. The claim that survives is
narrower: the MECHANISM is stable across all three (large, structured
Q-values; argmax decided by well under 1% of that magnitude; a
deterministic domain that turns "close" at one step into "wrong forever"),
the specific trajectory is not - two DIFFERENT trajectories, arrived at
independently, is still stronger evidence for that mechanism than one
exactly-repeatable instance would have been, but it is not evidence of a
frequency. Do not requote "0<->4" or "0.01-0.02" as an established number
independent of which run produced it.

Under epsilon-greedy TRAINING this matters concretely, not just in
principle - quantified, not just described, from the SECOND run's last 300
training episodes (`length` recovered as `diff(global_step)`):

| outcome (last 300 training episodes) | share |
|---|---|
| hits the full 100-step cap (the trap survives exploration too) | 43.7% |
| terminates early - falls in a hole | 54.7% |
| terminates early - reaches the goal | 1.7% |
| **terminates before the cap, total** | **56.3%** |

Exploration measurably escapes the trap the deterministic policy alone could
never leave - more than half of training episodes end at all only because of
it, not merely with more variance around a policy that already terminated.

**Why this belongs in the metric discussion, not just the results log:** in a
deterministic environment with sparse reward, exploration noise is not
merely adding variance around a real policy the way it does in exp01's
CartPole - it can be the only thing standing between the agent and an
infinite loop. Training return under epsilon-greedy therefore partly measures
EXPLORATION, not policy quality, more sharply here than in most RL
benchmarks. This is the mechanistic argument for why `greedy_best` must be
the primary reported statistic in this environment, with more reason than in
CartPole (not merely a preference, as the paragraph above already states) -
and a caution for reading any `best_ma`/`final_performance` number here
without its `greedy` counterpart alongside it.

**A methodological point this table's own numbers surface: greedy evaluation
in this environment has exactly ONE effective sample per checkpoint, not
`n_episodes`.** `FrozenLake-v1` here starts every episode at the same fixed
cell (state 0, the only `S` on the map) and is fully deterministic
(`is_slippery=False`); the greedy policy itself is a deterministic function
of the (fixed, evaluation-time) Q-network. `env.reset(seed=...)` therefore
cannot change the outcome - the seed feeds an RNG that a deterministic
environment with a fixed start never consults - so all `n_episodes` rollouts
at one checkpoint are the SAME trajectory, not `n_episodes` independent
draws. This is not inferred: it is exactly what the tables above already
show directly - all 10 (and, in the third run above, all 20) greedy-eval
episodes at a given checkpoint land on IDENTICAL lengths, every time this
has been checked. **Practical consequence: `greedy`'s reported CI is a CI
over N TRAINING seeds - the only axis of real variation - never over
`n_episodes`; averaging `n_episodes` identical values changes nothing about
the reported number, but reading `n_episodes=20` as 20 confirmations of
anything would overstate how much evidence one checkpoint carries.** This is
specific to FrozenLake's fixed start + determinism; exp01's CartPole DOES
randomise its initial state per `env.reset(seed=...)`, so the same 20/5
episodes there are genuinely independent draws and this caveat does not
transfer.

**Compute follow-up, not fixed here.** The consequence above means
`dqn/safe.py::evaluate_greedy`'s `n_episodes=20` (exp04's own
`eval_cfg_for`) spends 20 full rollouts per checkpoint to produce ONE
informative number, on every one of the ~10 checkpoints a 100k-step run
fires (`every_steps=10_000`) - roughly 200 rollouts per cell where 10 would
carry the identical information. Each rollout is cheap relative to a
training step (no gradient, `torch.no_grad()`, at most 100 env steps), so
this has not been worth interrupting a multi-day grid to fix, but it is real
waste, quantifiable, and `n_episodes=1` (or a check that short-circuits
after the first rollout on environments with no reset-time randomness) is
the natural fix - recorded in `docs/ROADMAP.md`, not implemented.

### Staging

Compute is spent only after the regime is shown alive — the failure that cost
exp01 two experiments.

- **Stage 0.** Gate notebook, capacity ladder, random-policy baseline,
  phantom-fraction prediction. Minutes.
- **Stage 1.** Three classical arms × FIX-01 on/off × 5 seeds. Cheap, and where
  the headline is decided. **Gate:** if `frozen_onehot_mlp` does not reach a
  success rate near 1.0, stop and repair the regime.
- **Stage 2.** Hybrid sweeps, FIX-01 on, 3 seeds coverage, then 8–10 robustness.

## 5. Hypotheses and readings

**H1 — DR depth (the exp03 replication).** Rising then saturating success rate
with `n_layers_q` replicates exp03's CartPole finding in a second environment and
generalises the thesis's one clean positive. Flat or falling bounds that claim to
CartPole — also a result, and a more interesting one than a second confirmation.
Falling with depth indicates optimisation difficulty, not expressivity, and
should not be called a barren plateau without gradient-norm evidence.

**H2 — embedding.** Binary-basis above scalar-to-phase at equal depth. Note this
is *not* a transfer claim: the chapter's single-seed Fig. 6 is too weak a
reference to transfer from. It is a first controlled measurement. The `noent`
ablation decides whether any gap is embedding or entanglement.

**H3 — FIX-01.** From stage 1. A large effect where the phantom fraction is ~10×
CartPole's, against the CartPole null, is the mechanism result. A null here too
means the bug is real but practically inconsequential — publishable, honest, and
it would substantially reduce how much weight the correction can carry.

**H4 — control, and the caveat that may swallow it.** If `frozen_matched_scalar`
reaches Config A, the circuit buys nothing over an equal-budget classical net on
the same information, and the honest claim concerns the *embedding*, not the
circuit. That would echo exp01.

But the smoke run (1 seed, 5k steps) suggests something more awkward: both scalar
arms sit at success 0.03-0.07 against a random floor of 0.015, while the one-hot
arm reaches 0.86-0.96. A network fed the raw state **index** has almost nothing to
work with - the ordering is row-major and carries no usable metric, so it must fit
a near-arbitrary 16-point function through one input.

If `frozen_matched_scalar` stays at chance, **H4 is unanswerable**: a hybrid
beating a dead control proves nothing, and claiming otherwise would repeat exactly
the error exp01 caught. Config A runs on that same scalar input and may be dead
for the same reason. In that case the informative comparison is Config B against
the one-hot MLP, and it must be reported as a result about the *encoding* rather
than dressed up as circuit-vs-classical.

> **UPDATE, 2026-08-21 — H4 confirmed unanswerable at L=1, with real statistical
> power.** Stage 1 finished at n=10 (not the 1-seed smoke this section was
> written against), and the result the smoke run warned about held up:
>
> | arm | success (MA-100, n=10) |
> |---|---|
> | `frozen_onehot_mlp` (liveness gate) | 0.999 ± 0.003 — **PASS**, comfortably |
> | `frozen_matched_scalar` (fair control) | 0.056–0.065 — chance (floor 0.015) |
> | `frozen_scalar_1q_L1` (Config A, L=1; 8/10 seeds) | 0.050 ± 0.009 — chance |
>
> Config A at depth 1 and its capacity-matched classical control are
> **statistically indistinguishable from each other and from a random policy**.
> This is not new information about the mechanism — it is the smoke-run warning
> confirmed, not contradicted — but it changes what H4 can be asked of *at this
> depth*: "does the circuit beat an equal-budget classical net" is unanswerable
> when both are dead, exactly as the paragraph above already said it would be.
>
> **UPDATE 2026-08-30 — the same table, recalculated with `final_performance`
> alongside the biased maximum.** A re-read of the same stage-1 manifests
> already on disk (n=10 each, `fix01=on`, no retraining) — `best_ma (max)` is
> the number the table above reports; `final_performance` is the mean over the
> last 10% of episodes, the unbiased counterpart (`docs/STATISTICS.md`):
>
> | arm | n | best_ma (max, biased) IQM (CI) | `final_performance` IQM (CI) |
> |---|---|---|---|
> | `frozen_onehot_mlp` (liveness gate) | 10 | 1.000 (0.998, 1.000) | 0.949 (0.946, 0.955) |
> | `frozen_matched_scalar` (fair control) | 10 | 0.052 (0.037, 0.087) | 0.001 (0.000, 0.013) |
> | `frozen_scalar_1q_L1` (Config A, L=1) | 10 | 0.048 (0.040, 0.057) | 0.000 (0.000, 0.000) |
>
> `frozen_onehot_mlp` barely moves (1.000 → 0.949): a genuinely alive,
> sustained policy, not a training-curve spike. `frozen_matched_scalar` and
> `frozen_scalar_1q_L1` collapse almost to zero (0.052 → 0.001, 0.048 → 0.000):
> the ~5% the biased maximum reported was itself mostly a lucky peak, not
> sustained performance — the "chance" reading in the paragraph above was, if
> anything, too generous. This sharpens H4's already-stated conclusion; it
> does not change it.
>
> **What changes going forward.** The useful question at Config A is no longer
> circuit-vs-classical; it is whether **depth (L = 5, 10, 15) rescues the scalar
> encoding by itself** — a question the dead L=1 point does not resolve one way
> or the other, since it says nothing about whether the network can learn to use
> a raw state index given more reuploading. **Config B vs `frozen_onehot_mlp`
> becomes the primary comparison** of this experiment's stage 2, not a fallback:
> it is the one pairing where the control is unambiguously alive. Stage 2 is run
> in full regardless — a rising Config-A curve would itself be informative (depth
> compensating for a bad encoding), and a flat one confirms the encoding, not the
> circuit, is the bottleneck, which is worth stating precisely rather than
> assuming. `frozen_scalar_1q_L1` is also short two seeds (1 and 2 of 10); that is
> tracked as a pending top-up, not a blocker.

**FIX-02 bonus.** Output scaling should be unnecessary here: with a 0/1 terminal
reward and γ = 0.99 the optimal action value is bounded by 1, while PauliZ
expectations already span [−1, 1]. On CartPole the same readout must reach
Q ≈ 100. If the hybrid learns here without scaling, the CartPole scaling
requirement is a *range* problem, not a representational one.

## 6. What would falsify the premise

Random-policy success above ~5% means the task is near-trivial and arms will not
separate. `frozen_onehot_mlp` failing means the regime is dead. Either invalidates
everything downstream — which is why stage 0 and the stage-1 gate exist and must
not be skipped.
