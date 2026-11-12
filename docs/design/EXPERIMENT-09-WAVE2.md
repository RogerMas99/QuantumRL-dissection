# Experiment 09 - wave 2 results (simulators)

> **SIMULATOR RESULTS - EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL**
> ([SIMULATED-RESULTS-POLICY.md](SIMULATED-RESULTS-POLICY.md)). The simulators were NOT accepted by the
> pre-registered validation rule (mean standardised difference +0.12, CI (+0.01, +0.25); see
> [EXPERIMENT-09-BATTERY.md](EXPERIMENT-09-BATTERY.md)): read only large effects. These results are used
> for one thing the policy allows: deciding which REAL-circuit runs are worth their hours.
> Generated 2026-10-21 by `analyze_wave2.py` and `experiments/exp09b_expressivity_diagnostics.py`.

### P - FrozenLake Config B, ENTANGLED (statevector), depth x Output Reuse (n=10)

| L | R | final_perf IQM (CI) | greedy_final IQM (CI) | alive | unentangled twin (su2) greedy / alive | P(ent>unent) final |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.000 (0.000-0.001) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 0.51 |
| 1 | 4 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 0.50 |
| 1 | 16 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 0.50 |
| 2 | 1 | 0.015 (0.001-0.057) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 0.75 |
| 2 | 4 | 0.016 (0.000-0.062) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 0.70 |
| 2 | 16 | 0.014 (0.001-0.047) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 0.80 |
| 3 | 1 | 0.056 (0.027-0.095) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 1.00 |
| 3 | 4 | 0.025 (0.011-0.069) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 0.94 |
| 3 | 16 | 0.046 (0.013-0.124) | 0.000 (0.000-0.000) | 0/10 | 0.000 / 0/10 | 0.94 |
| 5 | 1 | 0.942 (0.768-0.947) | 0.833 (0.500-1.000) | 9/10 | 0.000 / 0/10 | 1.00 |
| 5 | 4 | 0.513 (0.057-0.951) | 0.389 (0.000-0.889) | 5/10 | 0.000 / 0/10 | 1.00 |
| 5 | 16 | 0.372 (0.047-0.814) | 0.333 (0.000-0.833) | 4/10 | 0.000 / 0/10 | 1.00 |

Contrast: Output Reuse R=16 against R=1 on the entangled circuit

| L | greedy_final R16 / R1 | P(R16>R1) greedy | final_perf R16 / R1 | P(R16>R1) final | alive R16 / R1 |
|---|---|---|---|---|---|
| 2 | 0.000 / 0.000 | 0.50 | 0.014 / 0.015 | 0.49 | 0 / 0 |
| 3 | 0.000 / 0.000 | 0.50 | 0.046 / 0.056 | 0.44 | 0 / 0 |
| 5 | 0.333 / 0.833 | 0.31 | 0.372 / 0.942 | 0.34 | 4 / 9 |

### V / K - CartPole, 8 qubits, L=5: entangled (sv) against unentangled (su2), by Output Reuse R (n=10)

| R | entangled final_perf IQM (CI) | unentangled final_perf IQM (CI) | P(ent>unent) final / greedy / auc | informative note |
|---|---|---|---|---|
| 1 | 36.0 (28.2-45.5) | 43.7 (41.8-49.4) | 0.27 / 0.54 / 0.35 | |
| 4 | 99.0 (80.8-121.3) | 64.4 (51.9-74.0) | 0.90 / 0.80 / 1.00 | |
| 8 | 115.9 (95.0-127.6) | 73.5 (54.7-85.7) | 0.88 / 0.86 / 0.95 | |
| 16 | 119.4 (98.7-139.6) | 86.0 (73.5-100.4) | 0.82 / 0.85 / 1.00 | |
| 32 | 169.7 (138.0-192.3) | 75.9 (59.1-102.0) | 0.96 / 0.89 / 0.99 | |

Effect of Output Reuse itself, unentangled (su2) vs R=1: P(R>R1) final_perf

| R | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| P(OR>no OR) unentangled | 0.82 | 0.86 | 0.99 | 0.95 |
| P(OR>no OR) entangled | 0.99 | 1.00 | 0.99 | 1.00 |

### K - single-factor checks on CartPole (unentangled, n=10)

| factor | arm | final_perf IQM (CI) | greedy_final IQM | P(variant > baseline) final / greedy |
|---|---|---|---|---|
| baseline R=1 | su2 | 43.7 (41.8-49.4) | 53.6 | |
| output scale on, R=1 | su2 | 56.0 (42.9-78.7) | 65.0 | 0.65 / 0.55 |
| lr0.001, R=1 | su2 | 95.1 (78.7-109.8) | 111.6 | 0.99 / 0.87 |
| lr0.004, R=1 | su2 | 111.2 (93.0-121.5) | 123.7 | 0.99 / 0.88 |
| baseline R=16 | su2 | 86.0 (73.5-100.4) | 100.0 | |
| output scale on, R=16 | su2 | 83.6 (70.2-101.3) | 99.8 | 0.50 / 0.46 |
| lr0.001, R=16 | su2 | 111.5 (98.8-123.7) | 120.1 | 0.82 / 0.79 |
| lr0.004, R=16 | su2 | 102.0 (93.0-112.3) | 108.9 | 0.74 / 0.60 |

### H - FrozenLake axes never varied, one at a time (seeds 1-5; baseline = group F/G seeds 1-5)

| base arm | variant | final_perf IQM | greedy_final IQM | alive | P(variant>baseline) final |
|---|---|---|---|---|---|
| `su2_frozen_scalar_1q_L5` | baseline | 0.233 | 0.000 | 0/5 | |
| | head16 | 0.304 | 0.000 | 1/5 | 0.52 |
| | init0.1 | 0.088 | 0.000 | 0/5 | 0.32 |
| | init0.5 | 0.102 | 0.000 | 0/5 | 0.32 |
| | tis | 0.110 | 0.000 | 0/5 | 0.32 |
| | lr0.001 | 0.202 | 0.000 | 0/5 | 0.36 |
| | lr0.004 | 0.235 | 0.000 | 0/5 | 0.48 |
| `su2_frozen_scalar_1q_L10` | baseline | 0.178 | 0.000 | 0/5 | |
| | head16 | 0.361 | 0.000 | 1/5 | 0.60 |
| | init0.1 | 0.008 | 0.000 | 0/5 | 0.20 |
| | init0.5 | 0.010 | 0.000 | 0/5 | 0.16 |
| | tis | 0.010 | 0.000 | 0/5 | 0.20 |
| | lr0.001 | 0.223 | 0.000 | 0/5 | 0.48 |
| | lr0.004 | 0.143 | 0.000 | 0/5 | 0.48 |
| `su2_frozen_binary_4q_L5` | baseline | 0.000 | 0.000 | 0/5 | |
| | head16 | 0.953 | 1.000 | 5/5 | 1.00 |
| | init0.1 | 0.000 | 0.000 | 0/5 | 0.40 |
| | init0.5 | 0.000 | 0.000 | 0/5 | 0.40 |
| | tis | 0.000 | 0.000 | 0/5 | 0.48 |
| | lr0.001 | 0.001 | 0.000 | 0/5 | 0.64 |
| | lr0.004 | 0.000 | 0.000 | 0/5 | 0.40 |
---

# Diagnostics (no reinforcement learning)

## Part 1 - supervised fit of the optimal action values

| arm | params | lr | n | median MSE | argmax optimal (mean) | greedy reaches goal |
|---|---|---|---|---|---|---|
| `su2_frozen_scalar_1q_L1` | 10 | 0.01 | 5 | 1.33e-01 | 0.00 | 0 / 5 |
| `su2_frozen_scalar_1q_L1` | 10 | 0.05 | 5 | 1.33e-01 | 0.00 | 0 / 5 |
| `su2_frozen_scalar_1q_L5` | 18 | 0.01 | 5 | 8.52e-02 | 0.47 | 0 / 5 |
| `su2_frozen_scalar_1q_L5` | 18 | 0.05 | 5 | 7.87e-02 | 0.49 | 0 / 5 |
| `su2_frozen_scalar_1q_L10` | 28 | 0.01 | 5 | 7.69e-02 | 0.45 | 0 / 5 |
| `su2_frozen_scalar_1q_L10` | 28 | 0.05 | 5 | 7.70e-02 | 0.45 | 0 / 5 |
| `su2_frozen_scalar_1q_L15` | 38 | 0.01 | 5 | 7.69e-02 | 0.45 | 0 / 5 |
| `su2_frozen_scalar_1q_L15` | 38 | 0.05 | 5 | 7.70e-02 | 0.45 | 0 / 5 |
| `frozen_scalar_1q_fourier_ceiling_L5` | 44 | 0.01 | 5 | 3.12e-11 | 1.00 | 5 / 5 |
| `frozen_scalar_1q_fourier_ceiling_L5` | 44 | 0.05 | 5 | 4.25e-07 | 0.98 | 4 / 5 |
| `frozen_scalar_1q_fourier_ceiling_L10` | 84 | 0.01 | 5 | 5.31e-08 | 1.00 | 5 / 5 |
| `frozen_scalar_1q_fourier_ceiling_L10` | 84 | 0.05 | 5 | 2.38e-05 | 0.89 | 2 / 5 |
| `frozen_scalar_1q_fourier_ceiling_L15` | 124 | 0.01 | 5 | 6.99e-08 | 1.00 | 5 / 5 |
| `frozen_scalar_1q_fourier_ceiling_L15` | 124 | 0.05 | 5 | 3.78e-05 | 0.91 | 3 / 5 |
| `su2_frozen_binary_4q_L1` | 28 | 0.01 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `su2_frozen_binary_4q_L1` | 28 | 0.05 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `su2_frozen_binary_4q_L2` | 36 | 0.01 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `su2_frozen_binary_4q_L2` | 36 | 0.05 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `su2_frozen_binary_4q_L3` | 44 | 0.01 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `su2_frozen_binary_4q_L3` | 44 | 0.05 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `su2_frozen_binary_4q_L5` | 60 | 0.01 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `su2_frozen_binary_4q_L5` | 60 | 0.05 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `sv_frozen_binary_4q_L1` | 28 | 0.01 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `sv_frozen_binary_4q_L1` | 28 | 0.05 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `sv_frozen_binary_4q_L2` | 36 | 0.01 | 5 | 8.83e-02 | 0.27 | 0 / 5 |
| `sv_frozen_binary_4q_L2` | 36 | 0.05 | 5 | 8.83e-02 | 0.27 | 0 / 5 |
| `sv_frozen_binary_4q_L3` | 44 | 0.01 | 5 | 5.01e-03 | 0.33 | 0 / 5 |
| `sv_frozen_binary_4q_L3` | 44 | 0.05 | 5 | 3.29e-03 | 0.47 | 1 / 5 |
| `sv_frozen_binary_4q_L5` | 60 | 0.01 | 5 | 6.45e-08 | 1.00 | 5 / 5 |
| `sv_frozen_binary_4q_L5` | 60 | 0.05 | 5 | 7.24e-07 | 0.91 | 4 / 5 |
| `frozen_binary_4q_fourier_ceiling` | 20 | 0.01 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `frozen_binary_4q_fourier_ceiling` | 20 | 0.05 | 5 | 9.38e-02 | 0.27 | 0 / 5 |
| `frozen_matched_scalar` | 22 | 0.01 | 5 | 1.19e-01 | 0.11 | 0 / 5 |
| `frozen_matched_scalar` | 22 | 0.05 | 5 | 1.36e-01 | 0.05 | 0 / 5 |
| `frozen_scalar_mlp_large` | 4548 | 0.01 | 5 | 2.94e-05 | 0.85 | 2 / 5 |
| `frozen_scalar_mlp_large` | 4548 | 0.05 | 5 | 1.05e-01 | 0.56 | 0 / 5 |
| `frozen_onehot_mlp` | 5508 | 0.01 | 5 | 4.69e-10 | 0.96 | 4 / 5 |
| `frozen_onehot_mlp` | 5508 | 0.05 | 5 | 5.12e-13 | 1.00 | 5 / 5 |
| `frozen_binary_4q_L5` | 60 | 0.01 | 3 | 5.16e-06 | 1.00 | 3 / 3 |

## Part 2 - gradient variance at initialisation

| qubits | depth | kind | Var[dC/dtheta_0] | mean ||grad|| |
|---|---|---|---|---|
| 2 | 1 | unentangled (SU2) | 1.44e-02 | 0.090 |
| 2 | 2 | unentangled (SU2) | 6.70e-02 | 0.641 |
| 2 | 2 | entangled (statevector) | 6.70e-02 | 0.641 |
| 2 | 5 | unentangled (SU2) | 5.39e-02 | 0.877 |
| 2 | 5 | entangled (statevector) | 5.39e-02 | 0.877 |
| 2 | 10 | unentangled (SU2) | 4.76e-02 | 1.149 |
| 2 | 10 | entangled (statevector) | 4.76e-02 | 1.149 |
| 2 | 15 | unentangled (SU2) | 4.80e-02 | 1.324 |
| 2 | 15 | entangled (statevector) | 4.80e-02 | 1.324 |
| 4 | 1 | unentangled (SU2) | 1.66e-02 | 0.091 |
| 4 | 2 | unentangled (SU2) | 7.49e-02 | 0.664 |
| 4 | 2 | entangled (statevector) | 6.94e-02 | 0.354 |
| 4 | 5 | unentangled (SU2) | 5.76e-02 | 0.871 |
| 4 | 5 | entangled (statevector) | 2.54e-03 | 0.296 |
| 4 | 10 | unentangled (SU2) | 4.62e-02 | 1.140 |
| 4 | 10 | entangled (statevector) | 1.30e-03 | 0.399 |
| 4 | 15 | unentangled (SU2) | 3.42e-02 | 1.267 |
| 4 | 15 | entangled (statevector) | 1.14e-03 | 0.472 |
| 6 | 1 | unentangled (SU2) | 1.36e-02 | 0.086 |
| 6 | 2 | unentangled (SU2) | 6.04e-02 | 0.634 |
| 6 | 2 | entangled (statevector) | 5.49e-02 | 0.333 |
| 6 | 5 | unentangled (SU2) | 6.07e-02 | 0.878 |
| 6 | 5 | entangled (statevector) | 1.05e-03 | 0.166 |
| 6 | 10 | unentangled (SU2) | 4.49e-02 | 1.063 |
| 6 | 10 | entangled (statevector) | 4.36e-04 | 0.216 |
| 6 | 15 | unentangled (SU2) | 4.30e-02 | 1.273 |
| 6 | 15 | entangled (statevector) | 2.32e-04 | 0.273 |
| 8 | 1 | unentangled (SU2) | 1.59e-02 | 0.090 |
| 8 | 2 | unentangled (SU2) | 5.88e-02 | 0.671 |
| 8 | 2 | entangled (statevector) | 5.61e-02 | 0.341 |
| 8 | 5 | unentangled (SU2) | 6.16e-02 | 0.895 |
| 8 | 5 | entangled (statevector) | 7.69e-04 | 0.133 |
| 8 | 10 | unentangled (SU2) | 5.15e-02 | 1.135 |
| 8 | 10 | entangled (statevector) | 1.31e-04 | 0.128 |
| 8 | 15 | unentangled (SU2) | 4.08e-02 | 1.304 |
| 8 | 15 | entangled (statevector) | 6.17e-05 | 0.152 |
| 10 | 1 | unentangled (SU2) | 1.41e-02 | 0.082 |
| 10 | 2 | unentangled (SU2) | 7.12e-02 | 0.632 |
| 10 | 2 | entangled (statevector) | 6.89e-02 | 0.351 |
| 10 | 5 | unentangled (SU2) | 5.50e-02 | 0.872 |
| 10 | 5 | entangled (statevector) | 7.70e-04 | 0.130 |
| 10 | 10 | unentangled (SU2) | 5.40e-02 | 1.149 |
| 10 | 10 | entangled (statevector) | 1.02e-04 | 0.089 |
| 10 | 15 | unentangled (SU2) | 4.17e-02 | 1.293 |
| 10 | 15 | entangled (statevector) | 2.03e-05 | 0.086 |
