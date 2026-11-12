# Reconciliation of the number of runs

The thesis (table F.1) counts **6,368** runs. This table of results has **6368** rows. The count is exact; nothing was adjusted to make it agree.

## From the manifests of the original results tree to the table

| step | runs |
|---|---|
| manifests found in the results tree of the original repository (every folder, probes and duplicates included) | 6,377 |
| less: duplicated runs kept apart by the original repository (`_quarantine_dup_writers`, 8 runs excluded by the thesis, "eight duplicated runs") | -8 |
| less: repetition of one run of the reference control made to check reproducibility (not part of the campaign) | -1 |
| **runs of the campaign, as counted by the thesis** | **6,368** |
| of which: runs of the campaign with a manifest | 6,366 |
| of which: the two test runs of the terminal-handling probe (1,500 steps, seed 99; the thesis counts them in the rows of the output-reuse and depth experiments) | 2 |
| of which: the 10 runs of the reference control without reuse (added after the first campaign) | 10 (inside the 6,366) |
| of which: the pilot of 60,000 steps (config `pilot_60k`, no tables) | 18 (inside the 6,366) |

A previous count of 6,356 + 10 = 6,366 omitted the two probe runs; the thesis counts them, and so does this table.

## By configuration

| config | runs | curves in `results/` | raw archive only |
|---|---|---|---|
| additional_x100 | 2,300 | 2,300 | 0 |
| lr_experiment | 2,130 | 2,130 | 0 |
| pilot_60k | 18 | 18 | 0 |
| rq1_porting | 240 | 240 | 0 |
| rq2_capacity | 140 | 140 | 0 |
| rq3_depth | 111 | 110 | 1 |
| rq3_entanglement | 80 | 80 | 0 |
| rq3_output_reuse | 161 | 160 | 1 |
| rq4_ceilings | 100 | 60 | 40 |
| rq5_frozenlake | 110 | 110 | 0 |
| simulator_validation | 908 | 410 | 498 |
| terminal_ablation | 70 | 70 | 0 |
| **total** | **6,368** | **5,828** | **540** |

## By row of table F.1 of the thesis

| row of table F.1 | runs here | runs in the thesis | difference |
|---|---|---|---|
| Capacity ladder, CartPole (RQ2) | 128 | 128 | +0 |
| Output reuse (RQ3) | 131 | 131 | +0 |
| Reuploading depth, with and without entanglement (RQ3) | 61 | 61 | +0 |
| FrozenLake (RQ4, RQ5) | 230 | 230 | +0 |
| Configurations of the reference study (RQ1) | 240 | 240 | +0 |
| Ablations, crossing and ceilings (RQ3, RQ4) | 130 | 130 | +0 |
| Validation of the simulators | 888 | 888 | +0 |
| Additional results with 100 seeds | 2,300 | 2,300 | +0 |
| Classical twin and the superseded control | 130 | 130 | +0 |
| Learning-rate experiment | 2,130 | 2,130 | +0 |
| **total** | **6,368** | **6,368** | **+0** |


The sections that follow are produced by `python scripts/reconcile_runs.py --check`, which fails if a run belongs to no row of table F.1 or to two, if a row has a different number of runs than the thesis, or if the rows do not add up to the total. Every run counts, whether or not its curves are in `results/`: table F.1 is the cost of everything that was executed, discarded runs included.

## Runs by configuration and row of table F.1

| config | F.1-1 | F.1-2 | F.1-3 | F.1-4 | F.1-5 | F.1-6 | F.1-7 | F.1-8 | F.1-9 | F.1-10 | runs | of which curves in `results/` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| additional_x100 | - | - | - | - | - | - | - | 2300 | - | - | 2,300 | 2,300 |
| lr_experiment | - | - | - | - | - | - | - | - | - | 2130 | 2,130 | 2,130 |
| pilot_60k | 18 | - | - | - | - | - | - | - | - | - | 18 | 18 |
| rq1_porting | - | - | - | - | 240 | - | - | - | - | - | 240 | 240 |
| rq2_capacity | 40 | - | - | - | - | - | - | - | 100 | - | 140 | 140 |
| rq3_depth | - | - | 31 | - | - | 80 | - | - | - | - | 111 | 110 |
| rq3_entanglement | - | - | 30 | - | - | 50 | - | - | - | - | 80 | 80 |
| rq3_output_reuse | - | 131 | - | - | - | - | - | - | 30 | - | 161 | 160 |
| rq4_ceilings | 10 | - | - | 90 | - | - | - | - | - | - | 100 | 60 |
| rq5_frozenlake | - | - | - | 110 | - | - | - | - | - | - | 110 | 110 |
| simulator_validation | 20 | - | - | - | - | - | 888 | - | - | - | 908 | 410 |
| terminal_ablation | 40 | - | - | 30 | - | - | - | - | - | - | 70 | 70 |
| **total** | **128** | **131** | **61** | **230** | **240** | **130** | **888** | **2,300** | **130** | **2,130** | **6,368** | **5,828** |
| thesis, table F.1 | 128 | 131 | 61 | 230 | 240 | 130 | 888 | 2,300 | 130 | 2,130 | 6,368 | |

* F.1-1: Capacity ladder, CartPole (RQ2)
* F.1-2: Output reuse (RQ3)
* F.1-3: Reuploading depth, with and without entanglement (RQ3)
* F.1-4: FrozenLake (RQ4, RQ5)
* F.1-5: Configurations of the reference study (RQ1)
* F.1-6: Ablations, crossing and ceilings (RQ3, RQ4)
* F.1-7: Validation of the simulators
* F.1-8: Additional results with 100 seeds
* F.1-9: Classical twin and the superseded control
* F.1-10: Learning-rate experiment

## Hours of table F.1, unrounded (wall-clock seconds of every run, divided by 3,600)

| row | real circuit | simulators | classical | total | runs | of which not in `results/` |
|---|---|---|---|---|---|---|
| F.1-1 | 184.855 | 0.547 | 1.089 | 186.491 | 128 | 10 |
| F.1-2 | 356.284 | - | 1.240 | 357.525 | 131 | 1 |
| F.1-3 | 288.690 | - | - | 288.690 | 61 | 1 |
| F.1-4 | 1,067.160 | - | 6.659 | 1,073.819 | 230 | 40 |
| F.1-5 | 534.573 | - | 0.595 | 535.167 | 240 | 0 |
| F.1-6 | 671.127 | - | - | 671.127 | 130 | 0 |
| F.1-7 | - | 170.602 | - | 170.602 | 888 | 488 |
| F.1-8 | - | 580.137 | - | 580.137 | 2,300 | 0 |
| F.1-9 | - | - | 1.168 | 1.168 | 130 | 0 |
| F.1-10 | - | 217.034 | 45.536 | 262.570 | 2,130 | 0 |
| **total** | **3,102.689** | **968.320** | **56.287** | **4,127.296** | **6,368** | **540** |


Reading the first table: the 540 runs without curves in `results/` (`in_results` false in `runs.csv`) are 498 of `simulator_validation` (488 of the validation of the simulators, row F.1-7, and the 10 runs of the SU(2) emulator with the zero initialization that was superseded, row F.1-1), the 40 runs of the ceilings of FrozenLake with the CartPole hyperparameters (`rq4_ceilings`, row F.1-4) and one probe run in each of `rq3_depth` and `rq3_output_reuse`. A configuration is spread over several rows when its stages belong to different experiments: `rq2_capacity` between the capacity ladder (40 runs) and the classical twin (100), `rq4_ceilings` between the ceilings of CartPole (10) and of FrozenLake (90), and the same for the others as the table shows.

### The simulators cell of the capacity ladder

The cell holds 20 runs, all of the SU(2) emulator without entanglement: 10 with the zero initialization (`in_results` false) and 10 with the final initialization (`in_results` true). The sum of their wall-clock times is 1,969.5 s, that is **0.54708 h** (0.27358 h and 0.27350 h for the two groups); the table printed by this repository rounds it to 0.5. The thesis prints 0.6 in this cell. The count of runs (128) and the total of the column (968.3) coincide in both, so the difference is in this cell only; adding the two groups after rounding each to one decimal (0.3 + 0.3) would give 0.6, but how the printed value was obtained is not recorded.
