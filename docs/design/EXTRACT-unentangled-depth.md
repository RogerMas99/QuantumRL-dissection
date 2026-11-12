# Extract: design of the experiment on reuploading depth without entanglement

> **This is an extract, not a design document.** It is the module docstring of `experiments/exp03b_dqn_cartpole_dr_unentangled.py` of the
> original repository (`RogerMas99/qrl-dissection`, commit `18e124a`, 2026-08-05), copied literally. It is the only text of that experiment
> that states hypotheses before the runs (H1 and H2). The other experiments of the campaign that have no design document of their own
> (output reuse, depth, the configurations of the reference study, the crossing of depth and reuse) have no statement of expectations in
> the repository, and none is reproduced here.

```text

Experiment 03b - Data Reuploading under DQN, with entanglement held OFF.

WHY THIS EXISTS
---------------
exp03 found the thesis's one clean positive: monotone DR transfer under DQN on
CartPole, greedy 15 -> 35 -> 199 at depths L = 1/2/5. FIX-07 then showed that
result is confounded.

On the `skolik` template the entangling CZ ring is the last operation of every
layer, and CZ is diagonal, so it commutes with the PauliZ readout: the FINAL
layer's ring can never affect the output. Depth L therefore carries only L-1
EFFECTIVE entangling blocks. exp03 ran with `ent=True`, so its three depths
supplied 0, 1 and 4 effective entangling blocks - the depth axis moved
entanglement at the same time.

That leaves two readings of the same curve, and exp03 cannot separate them:

    (a) reuploading the input more often helps DQN, as it helps PPO;
    (b) entanglement coming online helps, and depth is along for the ride.

This experiment reruns the identical grid with `ent=False` throughout, so every
depth has zero entangling blocks and only reuploading varies.

    H1  the monotone rise survives -> reading (a). exp03's claim stands, now
        properly isolated, and the thesis keeps its clean positive.
    H2  the rise flattens or disappears -> reading (b). exp03's headline is
        really about entanglement, and must be rewritten. Better to find that
        here than in a viva.

Cost is the same as exp03 and the only change is one boolean, which is the
cheapest insurance available for the repo's most load-bearing result.

Comparison target: the paper's own `Skolik_DR_L{2,5}_Unentangled` runs, now in
`data/paper_ppo_summary.csv` at 10 seeds. Note their L1 unentangled cell is
vacuous for the same FIX-07 reason - it is a copy of the entangled circuit - so
depth 1 has no meaningful published counterpart.

Usage
-----
    python experiments/exp03b_dqn_cartpole_dr_unentangled.py --smoke
    python experiments/exp03b_dqn_cartpole_dr_unentangled.py
    python experiments/exp03b_dqn_cartpole_dr_unentangled.py --seeds 1 2 3 4 5 6 7 8 9 10

```
