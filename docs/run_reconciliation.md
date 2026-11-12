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
