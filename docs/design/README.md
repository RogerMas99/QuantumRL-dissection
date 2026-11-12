# Design documents

The documents in this folder are the **original design documents** of the experiments that had one, copied literally from the original
repository (`RogerMas99/qrl-dissection`, which stays public and archived and holds the history of the work). They
state the design, the predictions and the reading rule of each experiment before its runs. They use the vocabulary and the identifiers of the
time (experiment numbers, arm names, the numbering of the adjustments to the library); the table `docs/arm_names.csv` translates the arm
names, and `docs/adjustments.md` describes the adjustments.

| file | experiment |
|---|---|
| `EXPERIMENT-01.md` | capacity ladder (RQ2) |
| `EXPERIMENT-04.md` | FrozenLake (RQ4, RQ5) |
| `EXPERIMENT-09.md`, `EXPERIMENT-09-BATTERY.md`, `EXPERIMENT-09-WAVE2.md` | simulators and their validation against the real circuit |
| `EXPERIMENT-10-X100.md` | additional results with 100 seeds on the simulators |
| `EXPERIMENT-11-REPLICATION-CONTROL.md` | control that replicates its input (superseded) |
| `EXPERIMENT-12-CLASSICAL-TWIN.md` | classical twin of the hybrid agent |
| `EXPERIMENT-13-OR-VS-LR.md` | output reuse against the learning rate of the head |
| `EXTRACT-unentangled-depth.md` | **extract** (module docstring) with the two hypotheses of the experiment on depth without entanglement |

The other experiments (output reuse and depth of the main campaign, the 24 configurations of the reference study, the crossing of depth and
reuse) have no design document of their own.
