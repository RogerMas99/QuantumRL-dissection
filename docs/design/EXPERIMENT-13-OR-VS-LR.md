# Experiment 13 - Output Reuse against the learning rate of the head

Written 2026-11-06, **before any cell of this experiment has run**. Anything that changes afterwards is added at the
end under a dated "Amendments" heading, never edited above.

Two kinds of cells: classical arms on the real CartPole environment (agent T, and the PPO block D) and arms of the
hybrid on the batched statevector **emulator** (agent H, `agent_type` `sv`). The H cells are **simulated results**:
every manifest carries the `provenance` block and the policy of `docs/SIMULATED-RESULTS-POLICY.md` applies (the approval
recorded there for this experiment is the one of that table, row of 2026-11-06). Nothing here replaces any table of the
real circuit.

## 1. The proposition and what is tested

Let z be the feature vector that reaches the linear head and let Output Reuse replicate it R times:
q = [W_1 | ... | W_R][z; ...; z] + b.

- (i) The network depends only on V = sum_r W_r: the family of representable functions is the one without reuse.
- (ii) With SGD, momentum or Adam, **without global gradient-norm clipping and without weight decay**, training is
  identical to that of the network without reuse if (a) the learning rate of the weight matrix V of the head is
  multiplied by R and (b) V^0 = sum_r W_r^0. The learning rate of the bias does not change.
- Why: the R copies receive identical gradients, hence identical Adam moments and identical updates, and their sum
  moves R times as much. With PyTorch's default initialisation the variance of the entries of V is 1/(3f) for any R.
- The proposition does not depend on the RL algorithm (any loss, including PPO's clipped objective). What departs from
  its conditions is global gradient-norm clipping. SimplyQRL's PPO applies it (`ppo.py`: Adam with `eps=1e-5`, line 126;
  `clip_grad_norm_` with `max_grad_norm=0.5`, line 301). Even with clipping the R copies receive the same factor, so
  they stay identical and V moves R times faster: reuse R equals the network without reuse with the lr of V times R
  **provided its clipping uses the norm sqrt(R ||g_V||^2 + ||g_rest||^2)** instead of sqrt(||g_V||^2 + ||g_rest||^2).
  The equivalence is exact at the steps where clipping does not fire; where it fires, the two networks differ only by
  the common factor that multiplies every gradient, to which Adam is insensitive if it is constant in time.
- Our DQN uses Adam without clipping and without weight decay (checked in `src/simplyqrl/dqn.py`; there is no scheduler
  either), so it meets the conditions exactly.
- The reference study's classical control (replicated observation + linear layer) is covered too: the linear layer that
  reads the replicated input plays the role of V.

## 2. Predictions (fixed before launching)

- **P1.** "Collapsed" arm = reuse R. The collapsed arm is built by creating the model with reuse R for the seed, replacing
  its head by one with V^0 = sum_r W_r^0 and the same b^0, and training with the lr of V times R. In float64 with the same
  seed: the same trajectory until numerical chaos separates one greedy action; equivalent distributions across seeds.
- **P2.** "Fresh" arm (R = 1, default initialisation, lr of V times k) with k = R is approximately reuse R across seeds.
  This is the practical recipe: change the learning rate.
- **P3.** "Gradient x R" arm (R = 1, gradient of V times R, normal lr) is approximately R = 1. Adam is invariant to the
  scale of the gradient. This separates our proposition from the result of Detkov et al. (2023), which is for linear
  optimisers.
- **P4.** The performance curve against k of the fresh arm reproduces the curve against R; the sweep is extended beyond 32
  to locate where it degrades.
- **P5.** P1-P3 hold equally for the hybrid evaluated on the emulator, because the proposition does not depend on what
  produces z.
- **P6** (exploratory, no prediction). Global lr times k: indicates whether what matters is specifically the learning
  rate of the head.
- **P7.** Under PPO without clipping, reuse R is equivalent to lr x R (equivalence across seeds). With the clipping of 0.5
  the equivalence is approximate, and the separation must be the smaller the less the clipping fires.
- **P8.** With the modified clipping rule (norm computed with R ||g_V||^2), lr x R reproduces reuse R also with
  clipping: same trajectory in float64 until chaotic divergence and equivalence across seeds.

## 3. Agents and arms

**T**, the classical twin of exp12 (bounded variant, Tanh): `obs(4) -> Linear(4,6) -> ReLU -> Linear(6,8) -> Tanh ->
OutputReuse(R) -> Linear(8R,4) -> Identity -> Linear(4,2)` (`core/classical_twin.py`).
**H**, the hybrid of Fig. 4 (Skolik, 8 qubits, L = 5, entangled) with the circuit evaluated by the batched statevector
emulator (`core/statevector.py`, `agent_type` `sv`), same head. Never the PennyLane circuits. The H arms are, config by
config, the existing `sv_cartpole_L5` / `sv_cartpole_L5_OR{R}` arms (tested).

"head.weight" is the weight of the first top-level `nn.Linear` after the front end (the matrix that reads the
replicated features). Only that tensor carries the lr multiplier of the head; its bias, the second head layer and the
front end keep the base lr. No gradient clipping, weight decay or scheduler exists in the DQN pipeline (tested on the
source).

Arms per agent (exact keys of `ARMS`, prefix `classical_twin` for T and `hybrid_emu` for H):

| block | arm | R / k | what it is |
|---|---|---|---|
| A | `{P}_reuse_r{R}` | R in 1, 4, 8, 16, 32 | the model with reuse R (R = 1 is the common base; re-run here so the comparison is internal) |
| A | `{P}_collapsed_r{R}` | R in 4, 8, 16, 32 | reuse-R model, head replaced by V^0 = sum_r W_r^0, same b^0; lr of head.weight x R |
| A | `{P}_gradx{R}` | R in 8, 32 | R = 1; gradient of head.weight x R after `backward`, lr untouched |
| B | `{P}_headlr_x{k}` | k in 2, 4, 8, 16, 32, 64, 128 | fresh: R = 1, default initialisation, lr of head.weight x k |
| C | `{P}_globallr_x{k}` | k in 0.5, 2, 4, 8 (`0p5`) | R = 1, lr of all the parameters x k (exploratory) |

18 arms per agent in A+B (11 + 7), 22 with C. Block D (PPO) is in section 4.

**Target network (DQN).** The target network is rebuilt from the online network after the surgery (same structure and
weights) and then follows the usual hard copy every 500 steps.

**Implementation.** `core/lr_equivalence.py`; the vendored library is not touched. `SafeDQN` takes an optional
`post_build(dqn)` called once after `DQN()` and before `train()` (default `None`: unchanged behaviour). It performs the
surgery, builds Adam with the two parameter groups (same arguments as upstream), registers the gradient hook of the
"gradient x R" arm and attaches the diagnostics. Diagnostics every 1 000 steps (`runs/<run>_diag.csv`): `||V_eff||_F`
(for reuse, the norm of sum_r W_r), mean |Q| over the last 128 transitions of the replay buffer (both actions) and the
last TD loss that DQN logged (it logs every 100 steps). They draw from no RNG stream (tested: attaching them leaves the
training episodes identical).

## 4. Block D: PPO (second priority)

SimplyQRL's PPO, used **only for classical arms**, with the hyperparameters of the reference study for CartPole
(`PAPER_PPO_CONFIG` in `src/experiments/*.py` of the reference repository): lr 2.5e-4 annealed, 1 env, 512 steps,
gamma 0.99, GAE lambda 0.95, **16 minibatches** (the library default is 4), 4 epochs, clip 0.2, entropy 0.01, value
coefficient 0.5, `max_grad_norm` 0.5, `norm_adv`, `clip_vloss`, 100 000 steps, Adam `eps=1e-5`.

Agent: the reference study's classical control: observation reduced to indices [1, 2, 3], replicated R times, then one
linear layer for the actor and **one for the critic** (`net_arch = []`): the actor and the critic do not share the layer
that reads the replicated input, each has its own. The lr x R is applied to the weights (not the biases) of both.
The reduction of the observation is made by an observation wrapper (the agent receives 3 features), which is the same
function as upstream's `SelectiveOutputReuse` on the 4-dimensional observation (tested: identical initial weights under
one seed and identical outputs).

`qrl_dissection/ppo/lr_ppo.py::LRPPO` subclasses `simplyqrl.ppo.PPO`. Its `train` is the upstream loop with four
additions that do not change the behaviour at their defaults (tested: with the defaults the episode returns are those of
`PPO.train` bit for bit): lr by groups (the annealing multiplies every group; upstream only sets `param_groups[0]`),
`clip_mode` in {`torch` = upstream, `none`, `modified`}, a clipping diagnostic, and a greedy evaluation.

Arms (`experiments/exp13_ppo.py`; "lrx" = the same network without reuse, V^0 = sum_r W_r^0, lr of the V weights x R):

| arm | R | what |
|---|---|---|
| `ppo_reuse_clip_r{R}` | 1, 4, 8, 16, 32 | reuse R, clipping 0.5 (the reference study's case) |
| `ppo_reuse_noclip_r{R}` | 1, 4, 8, 16, 32 | reuse R, no clipping |
| `ppo_lrx_clip_r{R}` | 4, 8, 16, 32 | lr x R, clipping 0.5 with the usual norm |
| `ppo_lrx_noclip_r{R}` | 4, 8, 16, 32 | lr x R, no clipping |
| `ppo_lrx_modclip_r{R}` | 8, 32 | lr x R, clipping with the norm sqrt(R ||g_V||^2 + ||g_rest||^2) |

At R = 1 "lr x R" is the same network as reuse R = 1, so those cells are not duplicated (20 arms). The clipping
diagnostic is recorded at every update: the norm of the gradient before clipping, the factor min(1, 0.5 / (norm + 1e-6))
and whether the clipping fired (in the `none` arms, what the usual rule would have done). Per run the manifest stores the
fraction of clipped updates and the median factor, and the per-update array is saved. **That fraction, reported per arm,
is what decides how far the equivalence is from exact.**

PPO has no greedy evaluation in the library. Here: argmax of the actor on a separate environment, every 10 000 steps
(the evaluation runs at the end of the first iteration that reaches the multiple, so up to 511 steps later, and is
labelled with the multiple), 20 episodes, reset seeds `seed + 10000 + k` as in the DQN pipeline. It draws from no RNG
stream (tested).

## 5. Protocol

**DQN (A, B, C).** Identical to exp12: CartPole-v1, 100 000 steps, `batch_size` 128, `buffer_size` 10 000,
`train_frequency` 10, `learning_starts` 10 000, lr 2.5e-4, gamma 0.99, target network every 500 steps, epsilon 1.0 ->
0.05 over the first 50 %, terminal fix FIX-01 on. No hyperparameter is tuned: any negative result is "with these
hyperparameters". **Seeds 1-30** (1-10 are the seeds of exp12). **Greedy evaluation every 10 000 steps with 20 episodes**
(exp12 used 5; the internal comparison of this experiment is between arms with the same 20, so `greedy_final` has the
same noise across them; it has less noise than in exp12). Own folder `results/exp13_lr_equivalence/`, tag `exp13`.

**PPO (D).** As in section 4, seeds 1-30, own folder `results/exp13_ppo/`, tag `exp13`.

**Float64 traces (P1, P8).** Full training of the reuse-R and the collapsed-R arms with the whole network in float64
(inputs cast on entry), full protocol, no greedy evaluation, R in {8, 32}, 3 seeds, for T and for H (the emulator accepts
float64: tested), and for the PPO arms `ppo_reuse_clip_r{R}` against `ppo_lrx_modclip_r{R}`. For each pair: the first
environment step at which the action taken differs (or "never") and the maximum parameter difference up to that step
(effective parameters, with V = sum_r W_r for the reuse arm), traced every 100 steps.

## 6. Metrics, statistics, criteria

Per run: `final_perf` (mean of the last 10 % of the training episodes) and `greedy_final` (mean of the last 3 greedy
checkpoints). For PPO also the 50-episode moving average that the reference study uses (`best_ma50` there; the reference
reports 264-382 for its classical control). Per arm: IQM with a 95 % percentile bootstrap CI. Comparisons: P(a > b),
unpaired, ties 1/2, in both metrics. **No multiple-comparison correction.** "Strong" = P >= 0.95 in both metrics;
intermediate values "indicate a direction"; "mixed" if the two metrics point in opposite directions.

**Equivalence criterion (fixed beforehand, for P1-P3 and P7):** P(a > b) in [0.35, 0.65] in both metrics **and** the 95 %
CIs of the two IQMs overlap.

Contrasts:

- collapsed R against reuse R (R = 4, 8, 16, 32); fresh x R against reuse R (R = 4, 8, 16, 32); gradient x R against
  R = 1 (R = 8, 32);
- each fresh x k against R = 1; the best fresh x k against the best reuse R;
- C (global lr x k) against R = 1;
- D: each lr x R against reuse R **with the same clipping option**; reuse R with clipping against reuse R without it;
  the modified-clipping arm against reuse R with the usual clipping;
- for each R, the shift of the IQM (reuse minus fresh) with its bootstrap CI.

"Does not learn" = IQM <= 12 in both metrics.

Reading of the predictions (stated before the data): a prediction "holds" for an arm pair when the equivalence criterion
is met in both agents (P5 for the emulator, separately); it "does not hold" when the pair is strong or indicates a
direction in either sense; between the two (P in [0.25, 0.35) or (0.65, 0.75], or CIs that do not overlap with P inside
the band) it is reported as inconclusive, with the numbers. Any result that contradicts the proposition is reported with
its diagnosis; an implementation problem or a condition not foreseen is not hidden.

## 7. Tests (must all pass before launching)

`tests/test_exp13_dqn.py`, `tests/test_exp13_ppo.py` (results in the run record below): step-0 outputs of collapsed and
reuse below 1e-12 in float64 (T and H); after real DQN steps in float64 `max |V_collapsed - sum_r W_r|` below 1e-9 and
identical actions; the Adam moments of the R copies identical; the learning rates of every parameter group in every arm;
"gradient x R" and R = 1 produce Adam updates that differ only through `eps` at the first step; the target network has
the structure and the weights of the online one; the diagnostics change nothing; the `hybrid_emu_*` arms are flagged as
simulated; PPO: defaults identical to the vendored PPO, the wrapper control identical to the reference control, collapse
and RNG stream, annealing of every group, no-clipping and modified-clipping paths equal in float64 (and the usual rule
not), the clipping diagnostic.

## 8. Cost (estimates before launching; measured values go to the run record)

| block | cells | time per cell | core-hours | wall clock, 14 workers |
|---|---|---|---|---|
| T, A + B | 540 | ~40-50 s | ~6-8 | ~30-40 min |
| D, PPO | 600 | ~60-90 s | ~10-15 | ~45-65 min |
| H, A + B (emulator) | 540 | ~1 100 s | ~165 | ~12 h |
| T, C | 120 | ~40-50 s | ~1.5 | ~8 min |
| H, C | 120 | ~1 100 s | ~37 | ~2.6 h (not scheduled) |
| float64 traces | 12 + 6 pairs | T ~45 s, H ~1.5-2 x 1 100 s, PPO ~2 min per pair | ~5 | ~1-2 h in 2 workers |

Launch order (priority): A+B in T, D, A+B in H, C in T, C in H. C in H is not scheduled.

## Amendments

### 2026-11-07 - run record and deviations from the text above

* Seeds 1-30 in every block. Greedy evaluation: 20 episodes every 10 000 steps (the earlier experiments used fewer in some
  cases; when a table compares against runs of those, the comparison is across different evaluation sizes and is labelled).
* Upstream's epsilon-greedy draws its random actions from an action-space generator that no seed reaches. In every arm of
  this experiment the arm hook seeds it with the run seed (`core/lr_equivalence.py::make_post_build`). The exploration
  distribution is unchanged; cells become reproducible, which P1 needs. Consequence: the R = 1 / base arms of this
  experiment are not guaranteed to reproduce, seed for seed, the same arms of experiments that did not seed it.
* Block D: PPO hyper-parameters are those of the reference study's repository (16 minibatches, not the library default
  4); greedy evaluation is defined here (argmax of the actor, seeds seed + 10 000 + k); lr x R at R = 1 is the reuse R = 1
  arm, so block D has 600 cells (20 arms x 30 seeds), not 660. The vendored `ppo.py` is not edited: `LRPPO` subclasses it.
* Block C in H is not scheduled (cost, lowest priority).
* The learning-rate annealing of block D multiplies the base lr of EVERY parameter group (upstream only sets
  `param_groups[0]`, the only group it has). It was implemented so from the first launch, so no D arm had to be repeated.
  Tests `test_head_lr_is_R_times_the_others_at_every_update` and `test_reuse_arms_anneal_their_single_group` check, at
  every update, lr(head group) = R x lr(other group) and the decay. In DQN no scheduler or lr assignment exists
  (`grep` over `src/`: only `simplyqrl/ppo.py` touches `param_groups`; test `test_no_clipping_no_scheduler_in_dqn_source`).
* Tests at the time of launching: full suite 369 passed / 1 skipped; exp13 DQN 26, PPO 17 (+5 added by this amendment),
  blocks E-F-G 36.

### 2026-11-07 - addendum: blocks E, F, G (classical controls that contain Output Reuse)

Own folder `exp13_efg`, tag `exp13efg`, seeds 1-30, 100 000 steps, greedy 20 episodes. E, F use the CartPole DQN
hyper-parameters of the earlier controls (batch 128, buffer 10 000, train_frequency 10); G the FrozenLake ones (batch 128,
buffer 50 000, train_frequency 1, learning_starts 1 000). Implementation: `core/lr_efg.py`, `experiments/exp13_efg.py`,
`exp13_trace64_efg.py`, `exp13_analysis_efg.py`; tests `tests/test_exp13_efg.py`. No hyper-parameter was tuned.

* **E.** The additive Fourier ceiling feeds the 4 observables to the 8 wires cyclically (`idx = arange(8) % 4`), so wires i
  and i+4 have identical features: `Linear(80 -> 2)` (162 params) is Output Reuse R = 2 on `Linear(40 -> 2)` (82).
  Arms: `ceiling_dup` (162), `ceiling_unique` (82, base lr), `ceiling_unique_lrx2` (82, lr of the weight x 2, bias base, fresh
  initialisation), `ceiling_unique_lrx2_collapsed` (the dup model with the head collapsed: V0 = sum of the two blocks, same
  bias, lr x 2). Predictions: lrx2 ~ dup (both initialisations); ceiling_unique: none (its contrast with dup measures the
  effect of the x2). Also reported: P(arm > real hybrid with / without entanglement), 10 seeds of the REAL circuits.
* **F.** The capacity-matched control replicates the observation x4 before `Linear(16 -> 7)` (135 params): R = 4 on
  `Linear(4 -> 7)` (51). Arms: `matched_rep4_w7` (135), `matched_norep_w7_lrx4` (51, weight lr x 4, fresh), 
  `matched_norep_w7_lrx4_collapsed` (V0 = sum of the 4 blocks), `matched_norep_w18` (128), `matched_norep_w18_lrx4` (128).
  Prediction: norep_w7_lrx4 ~ rep4_w7. Also: P(real hybrid > arm) with the 10 seeds of `hybrid_fig4` (126 params), and the
  contrasts among the four arms of the addendum.
* **G.** `mlp_bits_w6` (58 params) and `mlp_bits_w12` (112) on the 4 bits in {0,1} of the state (least significant first, as
  the binary circuit's transformer, without its factor pi). Exploratory, no prediction. Reported: success rate (mean of the
  last 10 % of training episodes), seeds solved (greedy return 1 in all of the last 3 checkpoints), P against the real
  binary circuit with entanglement at L = 1 and L = 5.
* Decision recorded: the "lrx" arms of E and F start from the default initialisation of the small network (same variance as
  the sum of the blocks, different distribution); the `_collapsed` variants start from V0 = sum of the blocks, which is the
  initialisation the proposition requires. Both are run so that the equivalence test and the initialisation are separated.
* Float64 traces (E, F): pairs (ceiling_dup, ceiling_unique_lrx2_collapsed) and (matched_rep4_w7,
  matched_norep_w7_lrx4_collapsed), seeds 1-3, full run; reported: first step at which the action taken differs (with the
  exploration draws shared, the first greedy action that differs), parameter difference up to that step, Q difference of the
  final networks on 2 000 random states.
* Scheduling: blocks E, F, G run in parallel with H; no cell of H was interrupted. Estimated cost: E+F 270 cells x ~50 s,
  G 60 cells x ~5-6 min (one gradient step per environment step), about 9-10 core-hours.

## 2026-11-07 - control of the float64 PPO traces (sensitivity to a 1e-14 perturbation)

The float64 traces of block D (P7, P8) do not separate at R = 8 in 100 000 steps, but at R = 32 the two algebraically
equivalent runs end up apart (first differing action at 2.7e4-5.1e4 steps; parameter difference 2e-4 to 9e-3 before it). To
decide whether that is the training dynamics amplifying rounding error or a difference between the networks, the same arm
(`ppo_reuse_noclip_r{R}`, same seed) is run twice, the second time with every agent parameter shifted by a random 1e-14
before training (`experiments/exp13_trace64_ppo_perturb.py`; results in `exp13_traces/trace64_ppo_perturb.json`).

* R = 32, seeds 1-3: the perturbed copy separates at steps 26 688, 26 145 and 40 944, with parameter differences before the
  first differing action of 1.9e-3, 1.8e-2 and 3.9e-3. Same range as the equivalent pairs. R = 8: it does not separate
  (final differences 2e-9 to 2e-6), as the equivalent pairs do not.
* Reading: the separation of the equivalent pairs at R = 32 is the amplification of rounding error by the dynamics of this
  training, not evidence of a difference between the networks. The reference pair with the usual clipping rule, which is not
  equivalent, separates much earlier (573-2 600 steps) and by far more. The conclusions about P7 and P8 rest on the
  behavioural results (IQM equivalence), with the float64 trace as support at R = 8 only.
