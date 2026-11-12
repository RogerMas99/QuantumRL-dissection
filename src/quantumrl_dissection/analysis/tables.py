"""The result tables of the study, computed from ``results/``.

Every table is a function that takes ``Results`` and returns a ``Table``: a title, the header, and the rows as the strings that
the table prints. The statistics are those of the study:

    IQM            mean of the values sorted and trimmed by floor(0.25 n) at each end
    interval       percentile 2.5 / 97.5 of 10,000 bootstrap resamples of the IQM, ``default_rng(0)`` and one generator
                   per arm and statistic (training first, then greedy)
    P(a > b)       unpaired probability that a run of a beats a run of b, ties counting one half
    reading        "strong": P >= 0.95 (or <= 0.05) in both statistics; "level": P in [0.35, 0.65] in both; else "direction"

``scripts/make_tables.py`` writes them to ``tables/`` and, given the PDF of the thesis, checks every number against it.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantumrl_dissection.analysis import stats
from quantumrl_dissection.analysis.data import Results


@dataclasses.dataclass
class Table:
    id: str
    title: str
    header: List[str]
    rows: List[List[str]]
    pdf_page_hint: Optional[str] = None       # caption searched in the PDF ("Table D.3")
    note: str = ""

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.header)


# ---------------------------------------------------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------------------------------------------------
def iqm(a) -> float:
    return stats.iqm(np.asarray(a, dtype=float))


def interval(a, seed: int = 0) -> Tuple[float, float]:
    return stats.bootstrap_ci(np.asarray(a, dtype=float), seed=seed)


def fmt_ci(a, nd: int = 1, seed: int = 0) -> str:
    lo, hi = interval(a, seed)
    return f"{iqm(a):.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def P(a, b) -> float:
    return stats.probability_of_improvement(np.asarray(a, float), np.asarray(b, float))


def reading(p_train: float, p_greedy: float) -> str:
    if (p_train >= 0.95 and p_greedy >= 0.95) or (p_train <= 0.05 and p_greedy <= 0.05):
        return "strong"
    if 0.35 <= p_train <= 0.65 and 0.35 <= p_greedy <= 0.65:
        return "level"
    return "direction"


def n(x: int) -> str:
    return f"{x:,}"


def num_params(arm: str, env_id: Optional[str] = None) -> int:
    """Trainable parameters of an arm, measured by building it."""
    from quantumrl_dissection.agents.capacity import build_agent_for, count_trainable
    from quantumrl_dissection.arms import build_arm_config, environment_of
    env = env_id or environment_of(arm)
    at, cfg = build_arm_config(arm, env_id=env)
    return count_trainable(build_agent_for(at, cfg, env_id=env, is_qnet=True))


def _params_split(arm: str) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Total parameters and, for the arms that have a circuit or an extractor, its split with the head."""
    from quantumrl_dissection.agents.capacity import build_agent_for, count_trainable
    from quantumrl_dissection.arms import build_arm_config, environment_of
    env = environment_of(arm)
    at, cfg = build_arm_config(arm, env_id=env)
    ag = build_agent_for(at, cfg, env_id=env, is_qnet=True)
    total = count_trainable(ag)
    if at == "hybrid":
        head = count_trainable(ag.network[1:])
        return total, (total - head, head)
    if at == "classical_twin":
        return total, (count_trainable(ag.extractor), count_trainable(ag.head))
    return total, None


# ---------------------------------------------------------------------------------------------------------------------
# Selecting arms. ``S`` = (config, stage, arm)
# ---------------------------------------------------------------------------------------------------------------------
S = Tuple[str, str, str]


def vals(R: Results, s: S, metric: str) -> np.ndarray:
    return R.metric(s[0], s[1], s[2], metric)


def cell(R: Results, s: S, nd: int = 1) -> Tuple[str, str]:
    """(training, greedy) cells 'IQM (lo-hi)', one bootstrap generator per statistic as in the tables of the study."""
    f, g = vals(R, s, "final"), vals(R, s, "greedy")
    return fmt_ci(f, nd), fmt_ci(g, nd)


# Capacity ladder and reuse sweeps: (label, selector) -----------------------------------------------------------------------
RQ2, RQ3, RQ3D, RQ3E, RQ4, RQ5, RQ1, LR, TA = ("rq2_capacity", "rq3_output_reuse", "rq3_depth", "rq3_entanglement", "rq4_ceilings",
                                              "rq5_frozenlake", "rq1_porting", "lr_experiment", "terminal_ablation")

CAPACITY = [
    ("Reference control", (RQ2, "capacity_ladder", "reference_control")),
    ("Matched control", (RQ2, "capacity_ladder", "matched_control")),
    ("Hybrid agent", (RQ2, "capacity_ladder", "hybrid")),
    ("Unentangled hybrid agent", (RQ3E, "reuse_sweep_unentangled", "hybrid_unentangled")),
    ("Classical twin, bounded", (RQ2, "classical_twin", "classical_twin")),
    ("Classical twin, unbounded", (RQ2, "classical_twin", "classical_twin_unbounded")),
    ("Fourier ceiling", (RQ4, "cartpole_ceiling", "fourier_ceiling")),
    ("Large network", (RQ2, "capacity_ladder", "large_network")),
]
REUSE = (4, 8, 16, 32)


def reuse_selector(family: str, R: int) -> S:
    return {
        "hybrid": (RQ3, "reuse_sweep", f"hybrid_reuse{R}"),
        "hybrid_unentangled": (RQ3E, "reuse_sweep_unentangled", f"hybrid_unentangled_reuse{R}"),
        "reference": ((RQ3, "reference_control_without_reuse", "reference_control_reuse1") if R == 1 else
                      (RQ3, "reuse_sweep", "reference_control" if R == 4 else f"reference_control_reuse{R}")),
        "matched": (RQ3, "reuse_sweep", f"matched_control_reuse{R}"),
        "replicated": (RQ3, "replicated_control", f"matched_control_replicated{R}"),
        "twin": (RQ2, "classical_twin", "classical_twin" if R == 1 else f"classical_twin_reuse{R}"),
        "twin_unbounded": (RQ2, "classical_twin", "classical_twin_unbounded" if R == 1 else f"classical_twin_unbounded_reuse{R}"),
    }[family]


def depth_selector(entangled: bool, L: int) -> S:
    if entangled:
        return (RQ3D, "depth_sweep", "hybrid" if L == 5 else f"hybrid_depth{L}")
    return (RQ3E, "depth_sweep_unentangled", "hybrid_unentangled" if L == 5 else f"hybrid_unentangled_depth{L}")


# ---------------------------------------------------------------------------------------------------------------------
# Appendix D
# ---------------------------------------------------------------------------------------------------------------------
def table_D3(R: Results) -> Table:
    rows = []
    for label, s in CAPACITY:
        f, g = cell(R, s)
        rows.append([label, n(num_params(s[2])), f, g])
    return Table("D.3", "Results of the capacity ladder on CartPole: IQM and 95% interval over ten seeds",
                 ["Arm", "Par.", "Training return", "Greedy return"], rows, "Table D.3")


def table_D4(R: Results) -> Table:
    rows = []
    fam = [("Hybrid agent", "hybrid", REUSE), ("Unentangled hybrid agent", "hybrid_unentangled", REUSE),
           ("Reference control", "reference", (1,) + REUSE), ("Matched control", "matched", REUSE),
           ("Matched control replicating its input R times (superseded)", "replicated", (8, 16, 32)),
           ("Classical twin, bounded", "twin", REUSE), ("Classical twin, unbounded", "twin_unbounded", REUSE)]
    for label, key, rs in fam:
        for r in rs:
            s = reuse_selector(key, r)
            f, g = cell(R, s)
            rows.append([label, str(r), n(num_params(s[2])), f, g])
    return Table("D.4", "Results of the output-reuse sweep on CartPole: IQM and 95% interval over ten seeds",
                 ["Arm", "R", "Par.", "Training return", "Greedy return"], rows, "Table D.4")


def table_D5(R: Results) -> Table:
    rows = []
    for ent, label in ((True, "Entangled, no reuse"), (False, "Unentangled, no reuse")):
        for L in (1, 2, 5):
            s = depth_selector(ent, L)
            f, g = cell(R, s)
            rows.append([label, str(L), "-", n(num_params(s[2])), f, g])
    for L in (1, 2):
        for r in REUSE:
            s = (RQ3D, "depth_reuse_crossing", f"hybrid_depth{L}_reuse{r}")
            f, g = cell(R, s)
            rows.append([f"Entangled, L = {L}, with reuse", str(L), str(r), n(num_params(s[2])), f, g])
    return Table("D.5", "Results of the reuploading depth sweep and of its crossing with output reuse on CartPole",
                 ["Arm", "L", "R", "Par.", "Training return", "Greedy return"], rows, "Table D.5")


PORTING = [  # template, qubits, L, R, entangled, arm
    ("Skolik-derived", 4, 1, "-", "yes", "porting_skolik_4q_depth1"), ("Skolik-derived", 4, 2, "-", "yes", "porting_skolik_4q_depth2"),
    ("Skolik-derived", 4, 5, "-", "yes", "porting_skolik_4q_depth5"),
    ("Skolik-derived", 4, 2, "-", "no", "porting_skolik_4q_depth2_unentangled"),
    ("Skolik-derived", 4, 5, "-", "no", "porting_skolik_4q_depth5_unentangled"),
    ("Skolik-derived", 8, 1, "-", "yes", "porting_skolik_8q_depth1"), ("Skolik-derived", 8, 2, "-", "yes", "porting_skolik_8q_depth2"),
    ("Skolik-derived", 8, 5, "-", "yes", "porting_skolik_8q_depth5"),
    ("Perez-Salinas-derived", 1, 1, "-", "no effect", "porting_salinas_1q_depth1"),
    ("Perez-Salinas-derived", 1, 2, "-", "no effect", "porting_salinas_1q_depth2"),
    ("Perez-Salinas-derived", 1, 5, "-", "no effect", "porting_salinas_1q_depth5"),
    ("Perez-Salinas-derived", 2, 1, "-", "yes", "porting_salinas_2q_depth1"),
    ("Perez-Salinas-derived", 2, 2, "-", "yes", "porting_salinas_2q_depth2"),
    ("Perez-Salinas-derived", 2, 5, "-", "yes", "porting_salinas_2q_depth5"),
    ("Hsiao-derived", 4, 1, "4", "no", "porting_hsiao_reuse4"), ("Hsiao-derived", 4, 1, "8", "no", "porting_hsiao_reuse8"),
    ("Hsiao-derived", 4, 1, "16", "no", "porting_hsiao_reuse16"), ("Hsiao-derived", 4, 1, "32", "no", "porting_hsiao_reuse32"),
    ("Hsiao-derived", 4, 1, "4", "yes", "porting_hsiao_reuse4_entangled"),
    ("Hsiao-derived", 4, 1, "16", "yes", "porting_hsiao_reuse16_entangled"),
    ("Classical control", "-", "-", "4", "-", "reference_control"), ("Classical control", "-", "-", "8", "-", "reference_control_reuse8"),
    ("Classical control", "-", "-", "16", "-", "reference_control_reuse16"),
    ("Classical control", "-", "-", "32", "-", "reference_control_reuse32"),
]


def table_D6(R: Results) -> Table:
    rows = []
    for tmpl, q, L, r, ent, arm in PORTING:
        s = (RQ1, "porting", arm)
        f = vals(R, s, "final")
        g = vals(R, s, "greedy")
        rows.append([tmpl, str(q), str(L), r, ent, n(num_params(arm)), f"{iqm(f):.1f}", f"{iqm(g):.1f}"])
    return Table("D.6", "The 24 configurations of the reference study run in RQ1, with measured parameter counts and IQM over ten seeds",
                 ["Template", "Qubits", "L", "R", "Ent.", "Par.", "Training return (IQM)", "Greedy return (IQM)"], rows, "Table D.6")


def table_A2(R: Results) -> Table:
    rows = []
    for env, label, arm, stage_on, stage_off, nd in [
        ("CartPole", "Hybrid agent", "hybrid", "capacity_ladder", "capacity_ladder_no_terminal_adjustment", 1),
        ("CartPole", "Reference control", "reference_control", "capacity_ladder", "capacity_ladder_no_terminal_adjustment", 1),
        ("CartPole", "Matched control", "matched_control", "capacity_ladder", "capacity_ladder_no_terminal_adjustment", 1),
        ("CartPole", "Large network", "large_network", "capacity_ladder", "capacity_ladder_no_terminal_adjustment", 1),
        ("FrozenLake", "One-hot reference", "frozenlake_onehot_reference", "frozenlake_arms", "frozenlake_arms_no_terminal_adjustment", 3),
        ("FrozenLake", "Matched control", "frozenlake_matched_control", "frozenlake_arms", "frozenlake_arms_no_terminal_adjustment", 3),
        ("FrozenLake", "Large network, scalar input", "frozenlake_large_network_scalar", "frozenlake_arms",
         "frozenlake_arms_no_terminal_adjustment", 3)]:
        on, off = vals(R, (RQ2 if env == "CartPole" else RQ5, stage_on, arm), "final"), vals(R, (TA, stage_off, arm + "_no_terminal_adjustment"), "final")
        ong = vals(R, (RQ2 if env == "CartPole" else RQ5, stage_on, arm), "greedy")
        offg = vals(R, (TA, stage_off, arm + "_no_terminal_adjustment"), "greedy")
        rows.append([env, label] + [f"{iqm(x):.{nd}f}" for x in (on, off, ong, offg)])
    return Table("A.2", "Effect of the adjustment of terminal handling: IQM over ten seeds with and without it",
                 ["Environment", "Arm", "Training with", "Training without", "Greedy with", "Greedy without"], rows, "Table A.2")


# ---------------------------------------------------------------------------------------------------------------------
# Chapter 4
# ---------------------------------------------------------------------------------------------------------------------
def table_42(R: Results) -> Table:
    hy = (RQ2, "capacity_ladder", "hybrid")
    rows = []
    for label, s in [("Reference control", (RQ2, "capacity_ladder", "reference_control")),
                     ("Matched control", (RQ2, "capacity_ladder", "matched_control")),
                     ("Network without replication", (LR, "matched_control_with_reuse", "network_no_replication_w18")),
                     ("Classical twin, bounded", (RQ2, "classical_twin", "classical_twin")),
                     ("Classical twin, unbounded", (RQ2, "classical_twin", "classical_twin_unbounded"))]:
        pt, pg = P(vals(R, hy, "final"), vals(R, s, "final")), P(vals(R, hy, "greedy"), vals(R, s, "greedy"))
        rows.append([label, n(num_params(s[2])), f"{iqm(vals(R, s, 'final')):.1f}", f"{pt:.2f}", f"{pg:.2f}", reading(pt, pg)])
    return Table("4.2", "The hybrid agent against the classical controls on CartPole",
                 ["Classical arm", "Par.", "Training return", "P training", "P greedy", "Reading"], rows, "Table 4.2")


def table_44(R: Results) -> Table:
    rows = []
    items = [("Hybrid agent, L = 5, no reuse", (RQ2, "capacity_ladder", "hybrid"), (RQ3E, "reuse_sweep_unentangled", "hybrid_unentangled"))]
    for L in (1, 2, 5):
        items.append((f"Depth L = {L}", depth_selector(True, L), depth_selector(False, L)))
    for r in REUSE:
        items.append((f"Output reuse, R = {r}", reuse_selector("hybrid", r), reuse_selector("hybrid_unentangled", r)))
    for label, a, b in items:
        pt, pg = P(vals(R, a, "final"), vals(R, b, "final")), P(vals(R, a, "greedy"), vals(R, b, "greedy"))
        rows.append([label, f"{iqm(vals(R, a, 'final')):.1f}", f"{iqm(vals(R, b, 'final')):.1f}", f"{pt:.2f}", f"{pg:.2f}"])
    return Table("4.4", "Entangled against unentangled circuits on CartPole",
                 ["Configuration", "Entangled", "Unentangled", "P training", "P greedy"], rows, "Table 4.4")


def table_45(R: Results) -> Table:
    rows = []
    for L in (1, 2, 5):
        none = depth_selector(True, L)
        row = [f"L = {L}", f"{iqm(vals(R, none, 'final')):.1f}"]
        for r in REUSE:
            s = reuse_selector("hybrid", r) if L == 5 else (RQ3D, "depth_reuse_crossing", f"hybrid_depth{L}_reuse{r}")
            row.append(f"{iqm(vals(R, s, 'final')):.1f}")
        rows.append(row)
    # the row of L = 5 without reuse is the hybrid agent of the capacity ladder in the text of the table
    rows[-1][1] = f"{iqm(vals(R, (RQ2, 'capacity_ladder', 'hybrid'), 'final')):.1f}"
    return Table("4.5", "Depth crossed with output reuse on CartPole: IQM of the training return",
                 ["Depth", "No reuse", "R = 4", "R = 8", "R = 16", "R = 32"], rows, "Table 4.5")


def table_46(R: Results) -> Table:
    unent = (RQ3E, "reuse_sweep_unentangled", "hybrid_unentangled")
    ent = (RQ2, "capacity_ladder", "hybrid")
    items = [
        ("CartPole", "Ceiling, duplicated > unentangled", (RQ4, "cartpole_ceiling", "fourier_ceiling"), unent, 1),
        ("CartPole", "Ceiling, distinct > unentangled", (LR, "ceiling_with_reuse", "fourier_ceiling_distinct"), unent, 1),
        ("CartPole", "Ceiling, distinct > entangled", (LR, "ceiling_with_reuse", "fourier_ceiling_distinct"), ent, 1),
        ("FrozenLake", "Scalar ceiling > scalar circuit, L = 10", (RQ4, "frozenlake_ceilings", "frozenlake_scalar_ceiling_depth10"),
         (RQ5, "frozenlake_arms", "frozenlake_scalar_depth10"), 3),
        ("FrozenLake", "Binary, entangled > unentangled", (RQ5, "frozenlake_arms", "frozenlake_binary_depth5"),
         (RQ5, "frozenlake_arms", "frozenlake_binary_unentangled_depth5"), 3),
        ("FrozenLake", "Binary, entangled > linear ceiling", (RQ5, "frozenlake_arms", "frozenlake_binary_depth5"),
         (RQ4, "frozenlake_ceilings", "frozenlake_binary_linear_ceiling"), 3)]
    rows = []
    for env, label, a, b, nd in items:
        pt, pg = P(vals(R, a, "final"), vals(R, b, "final")), P(vals(R, a, "greedy"), vals(R, b, "greedy"))
        rows.append([env, label, f"{iqm(vals(R, a, 'final')):.{nd}f}", f"{iqm(vals(R, b, 'final')):.{nd}f}", f"{pt:.2f}", f"{pg:.2f}",
                     reading(pt, pg)])
    return Table("4.6", "The circuits against their classical ceilings",
                 ["Environment", "Contrast a > b", "IQM a", "IQM b", "P training", "P greedy", "Reading"], rows, "Table 4.6")


def solved_last_checkpoint(R: Results, config: str, stage: str, arm: str) -> Tuple[int, int]:
    """Seeds whose last greedy checkpoint reaches the goal on every evaluation episode (score >= 0.999)."""
    import pathlib
    gr = pd.read_csv(pathlib.Path(R.results) / config / "greedy_eval.csv.gz", keep_default_na=False, na_values=[""])
    gr = gr[(gr.stage == stage) & (gr.arm == arm)]
    last = gr.sort_values("global_step").groupby("seed").tail(1)
    return int((last.greedy_return >= 0.999).sum()), int(len(last))


def table_47(R: Results) -> Table:
    arms = [
        ("Large network, one-hot input (liveness)", (RQ5, "frozenlake_arms", "frozenlake_onehot_reference")),
        ("Large network, scalar input", (RQ5, "frozenlake_arms", "frozenlake_large_network_scalar")),
        ("Matched control", (RQ5, "frozenlake_arms", "frozenlake_matched_control")),
        ("Scalar circuit, L = 1", (RQ5, "frozenlake_arms", "frozenlake_scalar_depth1")),
        ("Scalar circuit, L = 5", (RQ5, "frozenlake_arms", "frozenlake_scalar_depth5")),
        ("Scalar circuit, L = 10", (RQ5, "frozenlake_arms", "frozenlake_scalar_depth10")),
        ("Scalar circuit, L = 15", (RQ5, "frozenlake_arms", "frozenlake_scalar_depth15")),
        ("Binary circuit, entangled, L = 1", (RQ5, "frozenlake_arms", "frozenlake_binary_depth1")),
        ("Binary circuit, entangled, L = 5", (RQ5, "frozenlake_arms", "frozenlake_binary_depth5")),
        ("Binary circuit, unentangled, L = 1", (RQ5, "frozenlake_arms", "frozenlake_binary_unentangled_depth1")),
        ("Binary circuit, unentangled, L = 5", (RQ5, "frozenlake_arms", "frozenlake_binary_unentangled_depth5")),
        ("Linear ceiling of the binary circuit", (RQ4, "frozenlake_ceilings", "frozenlake_binary_linear_ceiling")),
        ("Ceiling of the scalar circuit, L = 1", (RQ4, "frozenlake_ceilings", "frozenlake_scalar_ceiling_depth1")),
        ("Ceiling of the scalar circuit, L = 5", (RQ4, "frozenlake_ceilings", "frozenlake_scalar_ceiling_depth5")),
        ("Ceiling of the scalar circuit, L = 10", (RQ4, "frozenlake_ceilings", "frozenlake_scalar_ceiling_depth10")),
        ("Ceiling of the scalar circuit, L = 15", (RQ4, "frozenlake_ceilings", "frozenlake_scalar_ceiling_depth15")),
        ("Network on the bits, width 6 (exploratory)", (LR, "bits_networks", "frozenlake_bits_network_w6")),
        ("Network on the bits, width 12 (exploratory)", (LR, "bits_networks", "frozenlake_bits_network_w12"))]
    rows = []
    for label, s in arms:
        f, g = vals(R, s, "final"), vals(R, s, "greedy")
        lo, hi = interval(f)
        k, tot = solved_last_checkpoint(R, *s)
        rows.append([label, n(num_params(s[2])), f"{iqm(f):.3f}", f"{lo:.3f}-{hi:.3f}", f"{iqm(g):.3f}", f"{k}/{tot}"])
    return Table("4.7", "All FrozenLake arms: success rate during training (IQM and 95% interval), greedy success rate (IQM), seeds solved",
                 ["Arm", "Par.", "Training IQM", "95% interval", "Greedy", "Seeds solved"], rows, "Table 4.7")


# ---------------------------------------------------------------------------------------------------------------------
# Appendix E
# ---------------------------------------------------------------------------------------------------------------------
def _twin_arm(kind: str, k) -> str:
    return {"reuse": "classical_twin" if k == 1 else f"classical_twin_reuse{k}", "collapsed": f"classical_twin_collapsed{k}",
            "head": f"classical_twin_head_lr_x{k}", "grad": f"classical_twin_grad_x{k}", "all": f"classical_twin_all_lr_x{str(k).replace('.', 'p')}"}[kind]


def table_E1(R: Results) -> Table:
    rows = []
    layout = [("Reuse", "reuse", (1, 4, 8, 16, 32), "R = "), ("Collapsed start", "collapsed", (4, 8, 16, 32), "x"),
              ("Head", "head", (2, 4, 8, 16, 32, 64, 128), "x"), ("Gradient", "grad", (8, 32), "x"),
              ("All parameters", "all", (0.5, 2, 4, 8), "x")]
    for label, kind, ks, pre in layout:
        for k in ks:
            s = (LR, "dqn_twin", _twin_arm(kind, k))
            f, g = cell(R, s)
            rows.append([label, f"{pre}{k}", f, g])
    return Table("E.1", "Learning rate in place of reuse on the classical twin under DQN: IQM and 95% interval over thirty seeds",
                 ["Setting", "Factor", "Training return", "Greedy return"], rows, "Table E.1")


def table_E2(R: Results) -> Table:
    rows = []
    for r in (1, 4, 8, 16, 32):
        row = [str(r)]
        for clip in ("clip", "noclip"):
            for fam in ("reuse", "lr"):
                if fam == "reuse":
                    arm = f"ppo_reference_reuse{r}_{clip}"
                else:
                    arm = f"ppo_reference_lr_x{r}_{clip}"
                if r == 1 and fam == "lr":
                    row.append("-")
                    continue
                row.append(fmt_ci(vals(R, (LR, "ppo", arm), "best_ma50"), 1))
        rows.append(row)
    return Table("E.2", "Learning rate in place of reuse under PPO: best 50-episode moving average, IQM and 95% interval over thirty seeds",
                 ["R", "Clip, reuse", "Clip, LR", "No clip, reuse", "No clip, LR"], rows, "Table E.2")


def table_E3(R: Results) -> Table:
    rows = []
    items = [("CartPole", "Ceiling, duplicated features", (LR, "ceiling_with_reuse", "fourier_ceiling"), 1),
             ("CartPole", "Ceiling, distinct features", (LR, "ceiling_with_reuse", "fourier_ceiling_distinct"), 1),
             ("CartPole", "Ceiling, distinct features, LR 2", (LR, "ceiling_with_reuse", "fourier_ceiling_distinct_lr_x2"), 1),
             ("CartPole", "Matched control, width 7, input 4", (LR, "matched_control_with_reuse", "matched_control"), 1),
             ("CartPole", "Width 7, no replication, LR 4", (LR, "matched_control_with_reuse", "network_no_replication_w7_lr_x4"), 1),
             ("CartPole", "Width 18, no replication", (LR, "matched_control_with_reuse", "network_no_replication_w18"), 1),
             ("CartPole", "Width 18, no replication, LR 4", (LR, "matched_control_with_reuse", "network_no_replication_w18_lr_x4"), 1),
             ("FrozenLake", "Network on the bits, width 6", (LR, "bits_networks", "frozenlake_bits_network_w6"), 3),
             ("FrozenLake", "Network on the bits, width 12", (LR, "bits_networks", "frozenlake_bits_network_w12"), 3)]
    for env, label, s, nd in items:
        f, g = vals(R, s, "final"), vals(R, s, "greedy")
        rows.append([env, label, n(num_params(s[2])), fmt_ci(f, nd), fmt_ci(g, nd) if env == "CartPole" else f"{iqm(g):.3f}"])
    return Table("E.3", "The Fourier ceiling of CartPole on its distinct features, the network without replication and the networks on the bits",
                 ["Environment", "Arm", "Par.", "Training return", "Greedy return"], rows, "Table E.3")


def table_E4(R: Results) -> Table:
    rows = []
    items = [("CartPole", "Reference control", (RQ2, "capacity_ladder", "reference_control"), 1),
             ("CartPole", "Porting arm, classical, R = 16", (RQ1, "porting", "reference_control_reuse16"), 1),
             ("CartPole", "Porting arm, Hsiao-derived, R = 8", (RQ1, "porting", "porting_hsiao_reuse8"), 1),
             ("CartPole", "Hybrid agent", (RQ2, "capacity_ladder", "hybrid"), 1),
             ("CartPole", "Hybrid agent, R = 32", (RQ3, "reuse_sweep", "hybrid_reuse32"), 1),
             ("CartPole", "Large network", (RQ2, "capacity_ladder", "large_network"), 1),
             ("FrozenLake", "Linear ceiling of the binary circuit", (RQ4, "frozenlake_ceilings", "frozenlake_binary_linear_ceiling"), 3),
             ("FrozenLake", "Scalar circuit, L = 10", (RQ5, "frozenlake_arms", "frozenlake_scalar_depth10"), 3),
             ("FrozenLake", "Large network, scalar input", (RQ5, "frozenlake_arms", "frozenlake_large_network_scalar"), 3),
             ("FrozenLake", "Binary circuit, entangled, L = 5", (RQ5, "frozenlake_arms", "frozenlake_binary_depth5"), 3)]
    for env, label, s, nd in items:
        b = vals(R, s, "best_ma50")
        rows.append([env, label, f"{np.mean(b):.{nd}f} +- {np.std(b, ddof=1):.{nd}f}", f"{iqm(vals(R, s, 'final')):.{nd}f}",
                     f"{iqm(vals(R, s, 'greedy')):.{nd}f}"])
    return Table("E.4", "The statistic of the reference study, the best 50-episode moving average (mean and standard deviation over seeds), "
                 "against the two statistics of this work (IQM)", ["Environment", "Arm", "Best moving average", "Training return", "Greedy return"],
                 rows, "Table E.4")


# ---------------------------------------------------------------------------------------------------------------------
ALL: Dict[str, Callable[[Results], Table]] = {
    "A.2": table_A2, "4.2": table_42, "4.4": table_44, "4.5": table_45, "4.6": table_46, "4.7": table_47,
    "D.3": table_D3, "D.4": table_D4, "D.5": table_D5, "D.6": table_D6,
    "E.1": table_E1, "E.2": table_E2, "E.3": table_E3, "E.4": table_E4,
}
