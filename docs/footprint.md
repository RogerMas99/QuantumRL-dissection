# Energy and emissions: CodeCarbon calibration and estimate for the campaign

**None of the results changes.** The campaign was run without CodeCarbon. Here the consumption of representative runs is measured on the local machine and extrapolated to the wall-clock hours recorded in `results/runs.csv`. The calibration runs (tag `calib`) are written to a directory of their own (`calibration_runs/`, ignored by git), outside every experiment grid, and are not analysed as results.

Scripts: `scripts/footprint/` (`worker.py`, `calibrate.py`, `run_all.py`, `concurrency.py`, `summarize.py`). Derived tables: `docs/footprint/`.

**Unit.** A *process-hour* is one process running for one wall-clock hour. "kWh per process-hour" is the energy of the whole machine divided by (number of processes x wall-clock hours of the measurement); it is the quantity of appendix F.2 of the thesis ("energy per hour of process"). "Wall-clock hours" are plain elapsed hours, as in the `wall_seconds` column of `results/runs.csv`.

## 1. How CodeCarbon measures on this machine

CodeCarbon 3.3.2, `OfflineEmissionsTracker(country_iso_code="ESP", measure_power_secs=5)`, machine mode (it measures the whole machine). Machine: Windows 11, AMD Ryzen AI 7 350 (8 cores, 16 threads), 31 GB of RAM, **plus an NVIDIA GeForce RTX 5060 Laptop GPU** (PyTorch is CPU-only and does not use it, but CodeCarbon sees it and counts its idle consumption). Windows power plan "Balanced", plugged in. From its start-up log:

| component | method | measurement or estimate? |
|---|---|---|
| CPU | Windows Energy Meter Interface (RAPL-like package counters) | **measurement** (no TDP) |
| GPU | `pynvml` (NVML) | **measurement** (about 9 W at idle) |
| RAM | estimation model: 4 modules x 5 W (17-64 GB -> 4 modules) = a constant 20 W | **estimate** |

Carbon intensity of Spain: **0.17405 kg CO2-eq/kWh** (174.05 g/kWh), from the `global_energy_mix.json` file shipped with CodeCarbon (year 2023); according to its documentation it comes from Our World in Data / Ember (life cycle). Methodology: <https://docs.codecarbon.io/latest/explanation/methodology/>.

## 2. Design of the measurements

Six kinds of run, with the protocol of the corresponding stage of `configs/` (consecutive seeds until the time is up; the 100k steps are not touched). The configuration, stage and arm of each kind are listed in `scripts/footprint/worker.py`.

| kind | arm | environment |
|---|---|---|
| `real_cartpole` | `hybrid` (real circuit, 8 qubits, depth 5, entangled) | CartPole |
| `real_frozen` | `frozenlake_binary_depth5` (real circuit, binary encoding, entangled) | FrozenLake |
| `sv_cartpole` | `sim_statevector_hybrid` (batched state-vector simulator) | CartPole |
| `su2_cartpole` | `sim_emulator_hybrid_unentangled` (SU(2) emulator, hybrid without entanglement) | CartPole |
| `classic_dqn` | `classical_twin_reuse8` (classical twin with output reuse 8) | CartPole |
| `ppo` | `ppo_reference_reuse4_clip` (reference PPO control) | CartPole |

Two conditions per kind, 12 minutes each: (a) **one process**, `OMP_NUM_THREADS=1`; (b) **N processes in parallel** with a single tracker for the whole machine and the result divided by N: 14 for the simulators, the classical twin and PPO (the campaign used 14 processes per block), **8 for the two real circuits** (chosen because the number of processes of the local real-circuit campaign is not recorded). In addition, 2 minutes with the machine idle. No measurement came out anomalous, so none was repeated; the only adjustment was a preliminary smoke test of 40 s per kind, discarded, made while another analysis was running.

The calibration was measured with the scripts of the original working repository, which selected the same arms by their former names; `scripts/footprint/worker.py` selects them by the names of this repository (`docs/arm_names.csv` gives the correspondence) and with the protocol of the stages in `configs/`. The kinds are the same.

## 3. Calibration results

Duration measured by CodeCarbon (about 12.1 min per measurement). kWh = whole machine. "kWh per process-hour" = total kWh / (processes x wall-clock hours).

| kind | condition | processes | kWh total | CPU / RAM / GPU (kWh) | g CO2-eq | kWh per process-hour | mean machine power (W) | steps/s per process |
|---|---|---|---|---|---|---|---|---|
| idle | - | 0 | 0.0012 (2.0 min) | 0.0002 / 0.0007 / 0.0003 | 0.21 | - | **35.0** | - |
| `real_cartpole` | one process | 1 | 0.0086 | 0.0029 / 0.0040 / 0.0017 | 1.51 | 0.0431 | 43.1 | 18.4 |
| `real_cartpole` | parallel | 8 | 0.0151 | 0.0094 / 0.0040 / 0.0017 | 2.63 | 0.0094 | 75.2 | 17.4 |
| `real_frozen` | one process | 1 | 0.0083 | 0.0026 / 0.0040 / 0.0017 | 1.44 | 0.0413 | 41.3 | 2.5 |
| `real_frozen` | parallel | 8 | 0.0145 | 0.0088 / 0.0040 / 0.0017 | 2.53 | 0.0090 | 72.2 | 2.1 |
| `sv_cartpole` | one process | 1 | 0.0085 | 0.0028 / 0.0040 / 0.0017 | 1.47 | 0.0422 | 42.2 | 160.4 |
| `sv_cartpole` | parallel | 14 | 0.0146 | 0.0089 / 0.0040 / 0.0017 | 2.54 | 0.0052 | 72.5 | 84.7 |
| `su2_cartpole` | one process | 1 | 0.0108 | 0.0051 / 0.0040 / 0.0017 | 1.88 | 0.0537 | 53.7 | 1,327 |
| `su2_cartpole` | parallel | 14 | 0.0166 | 0.0108 / 0.0040 / 0.0017 | 2.89 | 0.0059 | 82.4 | 595 |
| `classic_dqn` | one process | 1 | 0.0114 | 0.0057 / 0.0040 / 0.0017 | 1.98 | 0.0566 | 56.6 | 4,836 |
| `classic_dqn` | parallel | 14 | 0.0166 | 0.0109 / 0.0040 / 0.0017 | 2.89 | 0.0059 | 82.4 | 2,115 |
| `ppo` | one process | 1 | 0.0092 | 0.0034 / 0.0040 / 0.0017 | 1.59 | 0.0456 | 45.6 | 1,730 |
| `ppo` | parallel | 14 | 0.0166 | 0.0109 / 0.0040 / 0.0017 | 2.89 | 0.0059 | 82.4 | 1,110 |

File: `docs/footprint/calibration_table.csv`. Readings:

- With the machine idle it draws **35 W** (RAM 20 W, GPU about 9 W, CPU about 8 W); one process adds only 6-22 W; with 14 processes the CPU reaches about 43-53 W (whole machine about 72-82 W). That is why the consumption **per process-hour** is 4.6 to 9.6 times lower in parallel than with one process: the base consumption is shared.
- The three 14-process conditions of `su2`, `classic_dqn` and `ppo` give the same total (0.0166 kWh, 53.5 W of CPU): the CPU is saturated and the consumption levels off.
- **Throughput against the campaign** (median of steps/wall_seconds in `runs.csv` for the local runs): the state-vector simulator 84.7 against 78 steps/s; the classical twin 2,115 against 1,970; the real FrozenLake circuit 2.1-2.5 against 2.3. They are comparable. PPO (1,110 against 873) and the SU(2) emulator (595 against 426) are 27 % to 40 % faster than in the campaign, which ran with more processes than threads. `real_cartpole` reads 17-18 steps/s because the first 10,000 steps (before training starts) run at more than 250 steps/s; the training throughput of the campaign is 5.5-8.5 steps/s. The consumption per hour does not depend on this, because the process works the whole time.

## 4. Extrapolation to the campaign

Hours = sum of `wall_seconds` of `results/runs.csv`: **6,368 runs, 4,127.3 h**, every run executed (the 540 runs without curves in `results/` included), split by where each run was made (`log_origin`): the local machine (5,624 runs, 2,043.5 h) or Colab (744 runs, 2,083.8 h). Tables: `docs/footprint/extrapolation_by_type.csv` and `extrapolation_totals.csv`. They are regenerated with

```bash
python scripts/footprint/summarize.py --calibration-table docs/footprint/calibration_table.csv --runs results/runs.csv --out <dir>
```

### 4.1 Hours on the local machine: calibrated measurement

To choose between one process and parallel, the number of runs alive at the same time was computed from the start time of each run and its `wall_seconds` (`scripts/footprint/concurrency.py`). Result:

| kind | wall-clock hours on this machine | simultaneous runs (mean weighted by hours) | condition used |
|---|---|---|---|
| real circuit (FrozenLake 1,006.5 h + CartPole 20.5 h) | 1,027.1 | 1.0 for FrozenLake (the whole 1,006.5 h with a single run alive); 2.0 for CartPole | **one process** |
| simulator `sv` | 861.6 | 13.7 | parallel (14) |
| simulator `su2` | 106.7 | 12.3 | parallel (14) |
| classical DQN | 28.6 | 17.8 | parallel (14) |
| PPO | 19.5 | 15.0 | parallel (14) |

The four local real-circuit CartPole runs overlapped on average with one other run; the one-process condition is kept for them, which is the one with the higher consumption per process-hour. Only runs listed in `runs.csv` are counted: not tests, traces or analyses.

| kind | wall-clock hours | kWh per process-hour used | kWh | kg CO2-eq (0.17405) |
|---|---|---|---|---|
| real circuit FrozenLake | 1,006.5 | 0.0413 | 41.55 | 7.23 |
| real circuit CartPole | 20.5 | 0.0431 | 0.88 | 0.15 |
| `sv` | 861.6 | 0.0052 | 4.46 | 0.78 |
| `su2` | 106.7 | 0.0059 | 0.63 | 0.11 |
| classical DQN | 28.6 | 0.0059 | 0.17 | 0.03 |
| PPO | 19.5 | 0.0059 | 0.11 | 0.02 |
| **Total, this machine** | **2,043.5** | | **47.80** | **8.32** |

89 % of that figure is the real circuit, which ran one run at a time: the machine was on with a single run, and almost all of its consumption is base consumption (RAM, idle GPU, idle CPU).

Variants of this part (all with the calibrated measurement):

- **Only the energy above the idle machine** (35 W, subtracting the base consumption): 9.3 kWh, **1.62 kg**. Lower bound, if the machine would have been on anyway.
- **CPU + RAM only, without the idle GPU**: 38.5 kWh, **6.70 kg**.
- **Everything CodeCarbon measures (whole machine, idle GPU included)**: 47.8 kWh, **8.32 kg**.

RAM is a CodeCarbon estimate (a constant 20 W), not a measurement; it accounts for 0.004 kWh of the 0.0085-0.017 kWh of each measurement.

### 4.2 Hours on Colab: assumption, not measurement

The Colab hardware is not recorded. An **explicit assumption** is used: power per process equal to the one measured here for the same kind (range: from the parallel condition to the one-process condition, 0.009-0.043 kWh per process-hour for the real circuit) and a grid intensity between **0.10 and 0.45 kg CO2-eq/kWh**.

| kind | wall-clock hours on Colab | kWh (low - high) | kg CO2-eq (low - high) |
|---|---|---|---|
| real circuit CartPole | 2,015.0 | 18.95 - 86.76 | 1.90 - 39.04 |
| real circuit FrozenLake | 60.6 | 0.55 - 2.50 | 0.05 - 1.13 |
| classical DQN | 8.2 | 0.05 - 0.46 | 0.00 - 0.21 |
| **Total, Colab** | **2,083.8** | **19.55 - 89.73** | **1.95 - 40.38** |

Sources of the intensity extremes: *Google Cloud, "Carbon data across Google Cloud regions"*, 2024 data (<https://github.com/GoogleCloudPlatform/region-carbon-info>; hourly Electricity Maps data, unaudited). Low end about 0.10: `europe-west1` (Belgium) 103.4 g/kWh and `us-west1` (Oregon) 79.2 g/kWh. High end about 0.45: `asia-northeast1` (Tokyo) 452.9 g/kWh and `us-central1` (Iowa) 412.7 g/kWh. Colab does not publish the region. Outside that range are the extremes of the world (2.7 g/kWh in `europe-north2`, 678.8 g/kWh in `asia-south1`). The power extremes (parallel and one process) span, roughly, from a server shared by many sessions to a virtual machine dedicated to a single one; which of the two applied is not known.

### 4.3 Total range

| part | basis | kWh | kg CO2-eq |
|---|---|---|---|
| this machine (2,043 h) | **calibrated measurement** (CodeCarbon, intensity of Spain) | 9.3 - 47.8 | 1.6 - 8.3 |
| Colab (2,084 h) | **assumption** (power from here, intensity 0.10-0.45) | 19.6 - 89.7 | 2.0 - 40.4 |
| **Whole campaign** (4,127 h) | | **28.9 - 137.5** | **3.6 - 48.7** |

Minimum = energy above the idle machine on this machine + low end on Colab. Maximum = everything measured on this machine + high end on Colab. The calibrated part contributes between 1.6 and 8.3 kg; the rest of the range (up to 40 kg) is the assumption about Colab. If on Colab the power per process were the one measured here in parallel and the grid 0.10, it would be about 2 kg; with the opposite extremes, about 40 kg.

## 5. Everyday equivalence

A new average EU car emitted 96.7 g CO2/km in 2025 (European Environment Agency, WLTP test; <https://www.eea.europa.eu/en/newsroom/news/average-co2-emissions-from-new-cars-and-vans-significantly-decreased-in-2025>; it is the figure for new cars, not for the fleet in circulation). With it:

- calibrated part, this machine (1.6 - 8.3 kg): **17 - 86 km** by car;
- whole campaign (3.6 - 48.7 kg): **37 - 504 km** by car.

## 6. Limits

- RAM is an estimate of the CodeCarbon model, not a measurement; the idle GPU (about 9 W) is in the total although it does no work.
- The power plan was "Balanced", on mains power; the campaign may have run with another. It is a system setting that was not touched.
- The 12 minutes of each measurement are a window: they do not cover the load variations of a hundred thousand steps (for example, the initial data-collection phase is shorter in the long runs).
- Concurrency is deduced from the start times and durations of the runs; there is no process log.
- The double-precision traces (7.5 h), tests, analyses and aborted runs are not included: they are not in `runs.csv`.
- The intensity of Spain is an annual mean for 2023; the real one depends on the hour and the year.
- The tables of `docs/footprint/` were derived from calibration runs kept outside the repository (`calibration_runs/`); `calibration_table.csv` is their summary, so the extrapolation can be regenerated without repeating the measurements.
