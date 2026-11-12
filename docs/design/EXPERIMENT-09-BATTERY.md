# Experiment 09 - validation battery of the simulators: results

> **SIMULATOR RESULTS - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL**
> ([SIMULATED-RESULTS-POLICY.md](SIMULATED-RESULTS-POLICY.md)). Design and acceptance rule:
> [EXPERIMENT-09.md](EXPERIMENT-09.md) section 7, fixed before any battery cell finished.
> Generated 2026-10-21 by `battery_report.py` from the manifests on disk.

**Verdict under the rule as written: NOT ACCEPTED, narrowly.** Criteria 2 and 3 pass; criterion 1
fails because the 95 % CI of the mean standardised difference is (+0.010, +0.250): it excludes 0
by 0.010, although the mean (+0.119) is well inside the 0.30 bound. The simulators run slightly
high on average against the real runs. Consequence stated in advance: nothing read from the
simulators is interpreted until the author decides.

**Update 2026-11-01:** re-evaluated with complete real references (n = 10 everywhere); verdict
unchanged, see the last section of this document ("Re-evaluation with complete real references").
The text above is the original evaluation and is kept as it was.

Note on method: a first run of this report used all 20 seeds of `sv_frozen_binary_4q_L5` and
gave criterion 1 PASS (mean d = +0.102, CI (-0.020, +0.243)); the rule is written for 10 seeds
per simulator arm, so the rule is evaluated with seeds 1-10 and the 20-seed pair is shown as an extra row.

> SIMULATOR RESULTS (agent types sv / su2) - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL (docs/SIMULATED-RESULTS-POLICY.md)

| kind | env | simulator arm | real arm | n sim / real | IQM final_perf sim / real | P(sim>real) final / greedy / auc | perm. p final | d | informative | note |
|---|---|---|---|---|---|---|---|---|---|---|
| sim-vs-real | CartPole | `sv_cartpole_L5` | `hybrid_fig4` | 10 / 10 | 35.989 / 35.801 | 0.53 / 0.62 / 0.50 | 0.85 | +0.07 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L5` | `hybrid_DR5` | 10 / 10 | 35.989 / 37.685 | 0.50 / 0.44 / 0.47 | 1.00 | -0.05 | yes | second reference (not in rule) |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR4` | `hybrid_OR4` | 10 / 10 | 99.010 / 94.660 | 0.53 / 0.53 / 0.54 | 0.85 | +0.13 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR8` | `hybrid_OR8` | 10 / 10 | 115.901 / 109.849 | 0.58 / 0.51 / 0.62 | 0.57 | +0.09 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR16` | `hybrid_OR16` | 10 / 10 | 119.379 / 104.209 | 0.66 / 0.53 / 0.55 | 0.26 | +0.49 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR32` | `hybrid_OR32` | 10 / 10 | 169.686 / 137.599 | 0.65 / 0.67 / 0.55 | 0.28 | +0.51 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L1` | `hybrid_DR1` | 10 / 10 | 10.319 / 10.137 | 0.60 / 0.72 / 0.67 | 0.46 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR4` | `hybrid_DR1_OR4` | 10 / 9 | 10.644 / 10.669 | 0.53 / 0.46 / 0.43 | 0.84 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR8` | `hybrid_DR1_OR8` | 10 / 9 | 11.509 / 11.224 | 0.62 / 0.39 / 0.53 | 0.40 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR16` | `hybrid_DR1_OR16` | 10 / 9 | 11.025 / 11.662 | 0.18 / 0.55 / 0.42 | 0.02 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR32` | `hybrid_DR1_OR32` | 10 / 9 | 10.615 / 10.751 | 0.49 / 0.64 / 0.49 | 0.97 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_cartpole_L2` | `hybrid_DR2` | 10 / 10 | 31.531 / 30.151 | 0.53 / 0.36 / 0.46 | 0.86 | +0.04 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR4` | `hybrid_DR2_OR4` | 10 / 9 | 46.511 / 46.275 | 0.47 / 0.43 / 0.48 | 0.83 | -0.04 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR8` | `hybrid_DR2_OR8` | 10 / 9 | 57.090 / 56.794 | 0.51 / 0.39 / 0.46 | 0.97 | -0.09 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR16` | `hybrid_DR2_OR16` | 10 / 9 | 73.532 / 70.519 | 0.54 / 0.53 / 0.48 | 0.78 | +0.23 | yes |  |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR32` | `hybrid_DR2_OR32` | 10 / 9 | 74.990 / 77.760 | 0.43 / 0.42 / 0.54 | 0.67 | -0.12 | yes |  |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L1` | `paper_skolik_4q_L1` | 10 / 7 | 9.624 / 9.634 | 0.46 / 0.55 / 0.37 | 0.80 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L2` | `paper_skolik_4q_L2` | 10 / 7 | 9.616 / 9.642 | 0.41 / 0.41 / 0.50 | 0.60 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L5` | `paper_skolik_4q_L5` | 10 / 5 | 9.624 / 9.627 | 0.54 / 0.53 / 0.44 | 0.87 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L1` | `paper_skolik_8q_L1` | 10 / 5 | 11.205 / 10.547 | 0.60 / 0.55 / 0.64 | 0.58 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L2` | `paper_skolik_8q_L2` | 10 / 4 | 13.392 / 10.194 | 0.82 / 0.62 / 0.80 | 0.07 | +1.05 | yes |  |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L5` | `paper_skolik_8q_L5` | 10 / 3 | 18.443 / 19.604 | 0.43 / 0.40 / 0.40 | 0.80 | -0.12 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L1` | `hybrid_DR1_noent` | 10 / 10 | 10.183 / 9.950 | 0.67 / 0.57 / 0.54 | 0.24 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `su2_cartpole_L2` | `hybrid_DR2_noent` | 10 / 10 | 36.112 / 34.547 | 0.54 / 0.72 / 0.56 | 0.79 | +0.22 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5 [init2pi]` | `hybrid_DR5_noent` | 10 / 10 | 43.673 / 41.789 | 0.66 / 0.59 / 0.52 | 0.23 | +0.10 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5 [init2pi]` | `hybrid_fig4_noent` | 10 / 9 | 43.673 / 42.528 | 0.59 / 0.58 / 0.63 | 0.54 | +0.05 | yes | second reference (not in rule) |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR4` | `hybrid_or_r4_noent` | 10 / 8 | 64.355 / 60.337 | 0.51 / 0.66 / 0.57 | 0.96 | +0.03 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR8` | `hybrid_or_r8_noent` | 10 / 9 | 73.461 / 68.745 | 0.56 / 0.50 / 0.52 | 0.74 | +0.04 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR16` | `hybrid_or_r16_noent` | 10 / 7 | 86.047 / 81.099 | 0.56 / 0.64 / 0.63 | 0.74 | +0.24 | yes |  |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR32` | `hybrid_or_r32_noent` | 10 / 5 | 75.852 / 86.655 | 0.36 / 0.42 / 0.48 | 0.44 | -0.26 | yes |  |
| sim-vs-real | CartPole | `su2_paper_skolik_4q_L2_noent` | `paper_skolik_4q_L2_noent` | 10 / 5 | 9.738 / 9.720 | 0.60 / 0.50 / 0.40 | 0.60 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | CartPole | `su2_paper_skolik_4q_L5_noent` | `paper_skolik_4q_L5_noent` | 10 / 4 | 9.679 / 9.662 | 0.45 / 0.46 / 0.60 | 0.83 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L1 (seeds<=10)` | `frozen_binary_4q_L1` | 10 / 10 | 0.000 / 0.000 | 0.55 / 0.50 / 0.56 | 1.00 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `su2_frozen_binary_4q_L1` | `frozen_binary_4q_noent_L1` | 10 / 10 | 0.000 / 0.000 | 0.49 / 0.50 / 0.50 | 1.00 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L5 (seeds<=10)` | `frozen_binary_4q_L5` | 10 / 10 | 0.942 / 0.952 | 0.21 / 0.43 / 0.48 | 0.03 | -0.10 | yes |  |
| sim-vs-real | FrozenLake | `su2_frozen_binary_4q_L5` | `frozen_binary_4q_noent_L5` | 10 / 10 | 0.000 / 0.000 | 0.55 / 0.50 / 0.58 | 1.00 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L5` | `frozen_scalar_1q_L5` | 10 / 10 | 0.102 / 0.125 | 0.49 / 0.50 / 0.53 | 0.99 | +0.10 | yes |  |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L10` | `frozen_scalar_1q_L10` | 10 / 10 | 0.046 / 0.081 | 0.56 / 0.45 / 0.45 | 0.67 | -0.12 | yes |  |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L15` | `frozen_scalar_1q_L15` | 10 / 10 | 0.010 / 0.011 | 0.46 / 0.50 / 0.47 | 0.81 |  | no | both dead: check 'both dead' OK |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L5` | `frozen_binary_4q_L5` | 20 / 10 | 0.928 / 0.952 | 0.27 / 0.40 / 0.41 | 0.04 | -0.45 | yes | extra, not in rule (all sv seeds) |
| twin | CartPole | `sv_cartpole_L5_noent` | `su2_cartpole_L5 [init2pi]` | 10 / 10 | 42.942 / 43.673 | 0.41 / 0.38 / 0.54 | 0.55 | -0.14 | yes |  |
| twin | FrozenLake | `sv_frozen_binary_4q_noent_L1` | `su2_frozen_binary_4q_L1` | 10 / 10 | 0.000 / 0.000 | 0.56 / 0.50 / 0.33 | 0.47 |  | no |  |
| twin | FrozenLake | `sv_frozen_binary_4q_noent_L5` | `su2_frozen_binary_4q_L5` | 10 / 10 | 0.000 / 0.000 | 0.45 / 0.50 / 0.54 | 1.00 |  | no |  |
| real-vs-real | CartPole | `hybrid_fig4` | `hybrid_DR5` | 10 / 10 | 35.801 / 37.685 | 0.49 / 0.34 / 0.48 | 0.97 | -0.11 | yes | noise floor (not in rule) |

**Criterion 1 (no global bias):** mean d = +0.119, 95% CI (+0.010, +0.250), 21 informative pairs -> FAIL
**Criterion 2 (no excess of disagreements):** 3 of 63 cells outside [0.25, 0.75] = 5% (limit 25%) -> PASS
**Criterion 3 (the two simulators agree):** 0 of 9 twin cells outside [0.25, 0.75] -> PASS
**Overall: NOT ACCEPTED** (still under docs/SIMULATED-RESULTS-POLICY.md in either case).

---

## Re-evaluation with complete real references (2026-11-01)

> SIMULATOR RESULTS - EXPLORATORY. The validation figures are the only approved use
> (SIMULATED-RESULTS-POLICY.md, approval of 2026-11-01); nothing else here is.

**Why.** On 2026-10-21 the real side of 10 of the 21 informative pairs had 3-9 seeds (for example
`paper_skolik_8q_L2` with n = 4, the pair with d = +1.05, and `hybrid_or_r32_noent` with n = 5). All
real reference arms now have n = 10 (exp07 and exp08 completed on 2026-11-01).

**What is unchanged.** The same pairs, the same definition of informative pair (IQM `final_perf` > 12 on
CartPole or > 0.05 on FrozenLake in at least one arm), the same simulator seeds 1-10, the same three
criteria and thresholds, the same RNG seed (0), the same code. The code is now in
`experiments/exp09c_battery_report.py` (it was a scratch script on 2026-10-21); pairs whose real
side did not change reproduce the earlier numbers exactly, which is the check that the script is the
same analysis. Second-reference pairs and the real-vs-real floor stay outside the rule. The original
evaluation above is kept as it was.

Command:

```
python experiments/exp09c_battery_report.py --results-root <results> --out <report.md> \
    --compare-with docs/EXPERIMENT-09-BATTERY.md
```

### Pairs, earlier and current

`n real old -> new` is the number of real seeds in the 2026-10-21 table and now. Pairs with no
`d` are not informative (both arms at the dead floor).

| kind | env | simulator arm | real arm | n sim | n real old -> new | IQM final sim / real (new) | P(sim>real) final / greedy / auc, old | P(sim>real) final / greedy / auc, new | d old -> new | informative old -> new |
|---|---|---|---|---|---|---|---|---|---|---|
| sim-vs-real | CartPole | `sv_cartpole_L5` | `hybrid_fig4` | 10 | 10 -> 10 | 35.989 / 35.801 | 0.53 / 0.62 / 0.50 | 0.53 / 0.62 / 0.50 | +0.07 -> +0.07 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L5` | `hybrid_DR5` | 10 | 10 -> 10 | 35.989 / 37.685 | 0.50 / 0.44 / 0.47 | 0.50 / 0.44 / 0.47 | -0.05 -> -0.05 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR4` | `hybrid_OR4` | 10 | 10 -> 10 | 99.010 / 94.660 | 0.53 / 0.53 / 0.54 | 0.53 / 0.53 / 0.54 | +0.13 -> +0.13 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR8` | `hybrid_OR8` | 10 | 10 -> 10 | 115.901 / 109.849 | 0.58 / 0.51 / 0.62 | 0.58 / 0.51 / 0.62 | +0.09 -> +0.09 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR16` | `hybrid_OR16` | 10 | 10 -> 10 | 119.379 / 104.209 | 0.66 / 0.53 / 0.55 | 0.66 / 0.53 / 0.55 | +0.49 -> +0.49 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L5_OR32` | `hybrid_OR32` | 10 | 10 -> 10 | 169.686 / 137.599 | 0.65 / 0.67 / 0.55 | 0.65 / 0.67 / 0.55 | +0.51 -> +0.51 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L1` | `hybrid_DR1` | 10 | 10 -> 10 | 10.319 / 10.137 | 0.60 / 0.72 / 0.67 | 0.60 / 0.72 / 0.67 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR4` | `hybrid_DR1_OR4` | 10 | 9 -> 10 | 10.644 / 10.647 | 0.53 / 0.46 / 0.43 | 0.52 / 0.49 / 0.45 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR8` | `hybrid_DR1_OR8` | 10 | 9 -> 10 | 11.509 / 11.184 | 0.62 / 0.39 / 0.53 | 0.63 / 0.42 / 0.51 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR16` | `hybrid_DR1_OR16` | 10 | 9 -> 10 | 11.025 / 11.589 | 0.18 / 0.55 / 0.42 | 0.18 / 0.52 / 0.45 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_cartpole_L1_OR32` | `hybrid_DR1_OR32` | 10 | 9 -> 10 | 10.615 / 10.830 | 0.49 / 0.64 / 0.49 | 0.46 / 0.57 / 0.45 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_cartpole_L2` | `hybrid_DR2` | 10 | 10 -> 10 | 31.531 / 30.151 | 0.53 / 0.36 / 0.46 | 0.53 / 0.36 / 0.46 | +0.04 -> +0.04 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR4` | `hybrid_DR2_OR4` | 10 | 9 -> 10 | 46.511 / 44.614 | 0.47 / 0.43 / 0.48 | 0.50 / 0.47 / 0.51 | -0.04 -> +0.03 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR8` | `hybrid_DR2_OR8` | 10 | 9 -> 10 | 57.090 / 54.551 | 0.51 / 0.39 / 0.46 | 0.54 / 0.45 / 0.50 | -0.09 -> +0.04 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR16` | `hybrid_DR2_OR16` | 10 | 9 -> 10 | 73.532 / 67.884 | 0.54 / 0.53 / 0.48 | 0.58 / 0.58 / 0.53 | +0.23 -> +0.38 | yes -> yes |
| sim-vs-real | CartPole | `sv_cartpole_L2_OR32` | `hybrid_DR2_OR32` | 10 | 9 -> 10 | 74.990 / 77.099 | 0.43 / 0.42 / 0.54 | 0.48 / 0.42 / 0.53 | -0.12 -> -0.03 | yes -> yes |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L1` | `paper_skolik_4q_L1` | 10 | 7 -> 10 | 9.624 / 9.634 | 0.46 / 0.55 / 0.37 | 0.44 / 0.51 / 0.36 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L2` | `paper_skolik_4q_L2` | 10 | 7 -> 10 | 9.616 / 9.621 | 0.41 / 0.41 / 0.50 | 0.43 / 0.50 / 0.43 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_paper_skolik_4q_L5` | `paper_skolik_4q_L5` | 10 | 5 -> 10 | 9.624 / 9.605 | 0.54 / 0.53 / 0.44 | 0.63 / 0.57 / 0.48 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L1` | `paper_skolik_8q_L1` | 10 | 5 -> 10 | 11.205 / 10.915 | 0.60 / 0.55 / 0.64 | 0.53 / 0.64 / 0.56 | - -> - | no -> no |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L2` | `paper_skolik_8q_L2` | 10 | 4 -> 10 | 13.392 / 12.944 | 0.82 / 0.62 / 0.80 | 0.56 / 0.64 / 0.58 | +1.05 -> +0.14 | yes -> yes |
| sim-vs-real | CartPole | `sv_paper_skolik_8q_L5` | `paper_skolik_8q_L5` | 10 | 3 -> 10 | 18.443 / 19.753 | 0.43 / 0.40 / 0.40 | 0.46 / 0.49 / 0.46 | -0.12 -> -0.36 | yes -> yes |
| sim-vs-real | CartPole | `su2_cartpole_L1` | `hybrid_DR1_noent` | 10 | 10 -> 10 | 10.183 / 9.950 | 0.67 / 0.57 / 0.54 | 0.67 / 0.57 / 0.54 | - -> - | no -> no |
| sim-vs-real | CartPole | `su2_cartpole_L2` | `hybrid_DR2_noent` | 10 | 10 -> 10 | 36.112 / 34.547 | 0.54 / 0.72 / 0.56 | 0.54 / 0.72 / 0.56 | +0.22 -> +0.22 | yes -> yes |
| sim-vs-real | CartPole | `su2_cartpole_L5 [init2pi]` | `hybrid_DR5_noent` | 10 | 10 -> 10 | 43.673 / 41.789 | 0.66 / 0.59 / 0.52 | 0.66 / 0.59 / 0.52 | +0.10 -> +0.10 | yes -> yes |
| sim-vs-real | CartPole | `su2_cartpole_L5 [init2pi]` | `hybrid_fig4_noent` | 10 | 9 -> 10 | 43.673 / 43.503 | 0.59 / 0.58 / 0.63 | 0.54 / 0.62 / 0.58 | +0.05 -> +0.02 | yes -> yes |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR4` | `hybrid_or_r4_noent` | 10 | 8 -> 10 | 64.355 / 60.434 | 0.51 / 0.66 / 0.57 | 0.52 / 0.65 / 0.59 | +0.03 -> +0.10 | yes -> yes |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR8` | `hybrid_or_r8_noent` | 10 | 9 -> 10 | 73.461 / 69.134 | 0.56 / 0.50 / 0.52 | 0.57 / 0.51 / 0.50 | +0.04 -> +0.04 | yes -> yes |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR16` | `hybrid_or_r16_noent` | 10 | 7 -> 10 | 86.047 / 89.465 | 0.56 / 0.64 / 0.63 | 0.45 / 0.47 / 0.64 | +0.24 -> -0.11 | yes -> yes |
| sim-vs-real | CartPole | `su2_cartpole_L5_OR32` | `hybrid_or_r32_noent` | 10 | 5 -> 10 | 75.852 / 77.880 | 0.36 / 0.42 / 0.48 | 0.46 / 0.53 / 0.58 | -0.26 -> +0.13 | yes -> yes |
| sim-vs-real | CartPole | `su2_paper_skolik_4q_L2_noent` | `paper_skolik_4q_L2_noent` | 10 | 5 -> 10 | 9.738 / 9.710 | 0.60 / 0.50 / 0.40 | 0.63 / 0.51 / 0.48 | - -> - | no -> no |
| sim-vs-real | CartPole | `su2_paper_skolik_4q_L5_noent` | `paper_skolik_4q_L5_noent` | 10 | 4 -> 10 | 9.679 / 9.693 | 0.45 / 0.46 / 0.60 | 0.47 / 0.52 / 0.57 | - -> - | no -> no |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L1 (seeds<=10)` | `frozen_binary_4q_L1` | 10 | 10 -> 10 | 0.000 / 0.000 | 0.55 / 0.50 / 0.56 | 0.55 / 0.50 / 0.56 | - -> - | no -> no |
| sim-vs-real | FrozenLake | `su2_frozen_binary_4q_L1` | `frozen_binary_4q_noent_L1` | 10 | 10 -> 10 | 0.000 / 0.000 | 0.49 / 0.50 / 0.50 | 0.49 / 0.50 / 0.50 | - -> - | no -> no |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L5 (seeds<=10)` | `frozen_binary_4q_L5` | 10 | 10 -> 10 | 0.942 / 0.952 | 0.21 / 0.43 / 0.48 | 0.21 / 0.43 / 0.48 | -0.10 -> -0.10 | yes -> yes |
| sim-vs-real | FrozenLake | `su2_frozen_binary_4q_L5` | `frozen_binary_4q_noent_L5` | 10 | 10 -> 10 | 0.000 / 0.000 | 0.55 / 0.50 / 0.58 | 0.55 / 0.50 / 0.58 | - -> - | no -> no |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L5` | `frozen_scalar_1q_L5` | 10 | 10 -> 10 | 0.102 / 0.125 | 0.49 / 0.50 / 0.53 | 0.49 / 0.50 / 0.53 | +0.10 -> +0.10 | yes -> yes |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L10` | `frozen_scalar_1q_L10` | 10 | 10 -> 10 | 0.046 / 0.081 | 0.56 / 0.45 / 0.45 | 0.56 / 0.45 / 0.45 | -0.12 -> -0.12 | yes -> yes |
| sim-vs-real | FrozenLake | `su2_frozen_scalar_1q_L15` | `frozen_scalar_1q_L15` | 10 | 10 -> 10 | 0.010 / 0.011 | 0.46 / 0.50 / 0.47 | 0.46 / 0.50 / 0.47 | - -> - | no -> no |
| sim-vs-real | FrozenLake | `sv_frozen_binary_4q_L5` | `frozen_binary_4q_L5` | 20 | 10 -> 10 | 0.928 / 0.952 | 0.27 / 0.40 / 0.41 | 0.27 / 0.40 / 0.41 | -0.45 -> -0.45 | yes -> yes |
| twin | CartPole | `sv_cartpole_L5_noent` | `su2_cartpole_L5 [init2pi]` | 10 | 10 -> 10 | 42.942 / 43.673 | 0.41 / 0.38 / 0.54 | 0.41 / 0.38 / 0.54 | -0.14 -> -0.14 | yes -> yes |
| twin | FrozenLake | `sv_frozen_binary_4q_noent_L1` | `su2_frozen_binary_4q_L1` | 10 | 10 -> 10 | 0.000 / 0.000 | 0.56 / 0.50 / 0.33 | 0.56 / 0.50 / 0.33 | - -> - | no -> no |
| twin | FrozenLake | `sv_frozen_binary_4q_noent_L5` | `su2_frozen_binary_4q_L5` | 10 | 10 -> 10 | 0.000 / 0.000 | 0.45 / 0.50 / 0.54 | 0.45 / 0.50 / 0.54 | - -> - | no -> no |
| real-vs-real | CartPole | `hybrid_fig4` | `hybrid_DR5` | 10 | 10 -> 10 | 35.801 / 37.685 | 0.49 / 0.34 / 0.48 | 0.49 / 0.34 / 0.48 | -0.11 -> -0.11 | yes -> yes |


### Verdict, before and now

| criterion | 2026-10-21 | 2026-11-01 (complete real references) |
|---|---|---|
| 1. no global bias (95 % bootstrap CI of mean d contains 0 and \|mean d\| <= 0.30) | mean d = +0.119, CI (+0.010, +0.250) -> **FAIL** | mean d = **+0.090**, CI (**+0.008, +0.175**) -> **FAIL** |
| 2. at most 25 % of cells outside P in [0.25, 0.75] | 3 of 63 = 5 % -> PASS | **1 of 63 = 2 %** -> PASS |
| 3. the two simulators agree (twins) | 0 of 9 -> PASS | 0 of 9 -> PASS (the twin pairs involve only simulator arms and real arms that were already complete, so they are unchanged) |
| **Overall** | NOT ACCEPTED | **NOT ACCEPTED** |

The informative flag did not change for any pair: the 21 informative pairs of 2026-10-21 are the
same 21 now. The definition decides that, and the dead-floor pairs stay dead with n = 10 real seeds.

**Noise floor (real vs real, `hybrid_fig4` of exp01 against `hybrid_DR5` of exp03):** d = -0.11, unchanged,
because both arms already had n = 10 on 2026-10-21.

### Reading (descriptive, no new criterion)

- The mean bias shrank (+0.119 -> +0.090) and the interval narrowed, but its lower end is still above 0
  (+0.008). Re-drawing the bootstrap with other RNG seeds moves it between +0.008 and +0.011, so the
  verdict does not hang on the RNG. A t-interval over the 21 d values is (-0.000, +0.181), that is,
  borderline; the rule as written is the bootstrap one and it fails. The rule was not relaxed.
- The pair that drove the old extreme, `paper_skolik_8q_L2` (d = +1.05 with n real = 4), is now +0.14
  with n real = 10. Complete references removed that outlier, as expected; they did not remove the
  small positive bias.
- What keeps the mean above 0 is a few entangled CartPole pairs: `sv_cartpole_L5_OR32` (d = +0.51),
  `sv_cartpole_L5_OR16` (+0.49) and `sv_cartpole_L2_OR16` (+0.38). Without these three the mean of the
  other 18 pairs is +0.03. These three pairs already had n real = 10 on 2026-10-21, so they are not an
  effect of the missing seeds; they say that the batched statevector simulator runs higher than the
  real circuit at high Output Reuse. This is a description of the 21 pairs, not a new criterion.
- Cells outside [0.25, 0.75] (pairs inside the rule): only `sv_frozen_binary_4q_L5 (seeds<=10)` vs
  `frozen_binary_4q_L5`, `final_perf`, P(sim > real) = 0.21 (same cell as before; n = 10 on both sides).
  The earlier cells outside the window that came from short real references no longer appear.

### Consequence

Under the rule as written the simulators are still **not accepted** as stand-ins for the real circuit
(criterion 1 fails by 0.008, narrower than before). The conclusion of the 2026-10-21 evaluation stands.
