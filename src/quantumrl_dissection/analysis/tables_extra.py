"""Further result tables: the validation of the simulators (C.1), the additional results with 100 seeds (E.5), the configurations of
the reference study by template (4.1), cost (F.1, F.2) and the structure of the CartPole arms (D.2). See ``tables`` for the
conventions."""
from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd

from quantumrl_dissection.analysis import stats
from quantumrl_dissection.analysis.data import Results
from quantumrl_dissection.analysis.tables import (
    ALL, LR, REUSE, RQ1, RQ2, RQ3, RQ4, RQ5, Table, _params_split, iqm, n, num_params, P, vals,
)


def _boot_p(a, b, B: int = 10_000, seed: int = 0) -> Tuple[float, float]:
    """Percentile interval of P(a > b) under bootstrap resampling of the seeds of both arms."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ps = np.empty(B)
    for i in range(B):
        ps[i] = stats.probability_of_improvement(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
    return float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))


def table_E5(R: Results) -> Table:
    h = "sim_statevector_hybrid"
    items = [("Entangled > unentangled, L = 5", "No reuse", h, h + "_unentangled", 1)]
    items += [("Entangled > unentangled, L = 5", f"R = {r}", f"{h}_reuse{r}", f"{h}_unentangled_reuse{r}", 1) for r in REUSE]
    items += [("Depth, no reuse", "Entangled, L = 2 > L = 1", f"{h}_depth2", f"{h}_depth1", 1),
              ("Depth, no reuse", "Entangled, L = 5 > L = 2", h, f"{h}_depth2", 1),
              ("Depth, no reuse", "Unentangled, L = 2 > L = 1", f"{h}_unentangled_depth2", f"{h}_unentangled_depth1", 1),
              ("Depth, no reuse", "Unentangled, L = 5 > L = 2", f"{h}_unentangled", f"{h}_unentangled_depth2", 1)]
    items += [("Depth crossed with reuse", f"L = 2, R = {r} > no reuse", f"{h}_depth2_reuse{r}", f"{h}_depth2", 1) for r in REUSE]
    items += [("Depth crossed with reuse", "R = 32, L = 5 > L = 2", f"{h}_reuse32", f"{h}_depth2_reuse32", 1),
              ("FrozenLake, binary, L = 5", "Entangled > unentangled", "sim_statevector_frozenlake_binary_depth5",
               "sim_statevector_frozenlake_binary_unentangled_depth5", 3)]
    rows = []
    for grp, label, a, b, nd in items:
        da, db = R.additional_x100(a), R.additional_x100(b)
        pt, pg = P(da["final"], db["final"]), P(da["greedy"].dropna(), db["greedy"].dropna())
        lo, hi = _boot_p(da["final"], db["final"])
        rows.append([grp, label, f"{iqm(da['final']):.{nd}f}", f"{iqm(db['final']):.{nd}f}", f"{pt:.2f}", f"{lo:.2f}-{hi:.2f}", f"{pg:.2f}"])
    return Table("E.5", "Additional contrasts computed with the batched state-vector simulator, 100 seeds per arm",
                 ["Group", "Contrast a > b", "IQM a", "IQM b", "P training", "95% interval", "P greedy"], rows, "Table E.5")


def table_C1(R: Results) -> Table:
    from quantumrl_dissection.analysis import validation
    _, v = validation.evaluate(R)
    rows = [["No global bias", f"d = {v['md']:+.3f} ({v['lo']:+.3f}, {v['hi']:+.3f}), {'met' if v['c1'] else 'not met'}"],
            ["No excess of disagreements", f"{v['outside']} of {v['cells']} outside, {'met' if v['c2'] else 'not met'}"],
            ["Agreement between simulators", f"{v['twin_bad']} of {v['twin_cells'] // 3} outside, {'met' if v['c3'] else 'not met'}"]]
    return Table("C.1", "Outcome of the validation of the simulators against real arms with ten seeds each", ["Condition", "Outcome"], rows, "Table C.1")


def table_41(R: Results) -> Table:
    from quantumrl_dissection.arms import REFERENCE_STUDY_RESULTS
    summ = pd.read_csv(pathlib.Path(R.results) / "reference_study" / "reference_study_ppo_summary.csv")
    ppo = {r.config: r.best_ma50_mean for r in summ.itertuples()}
    groups = [
        ("Classical control", "R in {4, 8, 16, 32}", [("reference_control" if r == 4 else f"reference_control_reuse{r}") for r in REUSE]),
        ("Hsiao, unentangled", "R in {4, 8, 16, 32}", [f"porting_hsiao_reuse{r}" for r in REUSE]),
        ("Hsiao, entangled", "R in {4, 16}", ["porting_hsiao_reuse4_entangled", "porting_hsiao_reuse16_entangled"]),
        ("Perez-Salinas, 1 qubit", "L in {1, 2, 5}", [f"porting_salinas_1q_depth{L}" for L in (1, 2, 5)]),
        ("Perez-Salinas, 2 qubits", "L in {1, 2, 5}", [f"porting_salinas_2q_depth{L}" for L in (1, 2, 5)]),
        ("Skolik, 4 qubits", "L in {1, 2, 5}", [f"porting_skolik_4q_depth{L}" for L in (1, 2, 5)]),
        ("Skolik, 4 qubits, unentangled", "L in {2, 5}", [f"porting_skolik_4q_depth{L}_unentangled" for L in (2, 5)]),
        ("Skolik, 8 qubits", "L = 1", ["porting_skolik_8q_depth1"]), ("Skolik, 8 qubits", "L = 2", ["porting_skolik_8q_depth2"]),
        ("Skolik, 8 qubits", "L = 5", ["porting_skolik_8q_depth5"])]

    def rng_s(x, nd):
        a, b = f"{min(x):.{nd}f}", f"{max(x):.{nd}f}"
        return a if a == b else f"{a}-{b}"
    rows = []
    for tmpl, sweep, arms in groups:
        par = [num_params(a) for a in arms]
        fin = [iqm(vals(R, (RQ1, "porting", a), "final")) for a in arms]
        ma = [float(np.mean(vals(R, (RQ1, "porting", a), "best_ma50"))) for a in arms]
        pp = [ppo[REFERENCE_STUDY_RESULTS[a]] for a in arms]
        pr = f"{min(par)}-{max(par)}" if min(par) != max(par) else str(par[0])
        rows.append([tmpl, sweep, pr, rng_s(fin, 1), rng_s(ma, 1), rng_s(pp, 1)])
    return Table("4.1", "The configurations of the reference study under DQN, by template, against their PPO counterparts",
                 ["Template", "Sweep", "Par.", "DQN training return", "DQN moving avg.", "PPO moving avg."], rows, "Table 4.1")


F1_ROWS = [  # row label, membership of a run by (configuration, stage, variant)
    ("Capacity ladder, CartPole (RQ2)", lambda c, s, v: (c, s) in {("rq2_capacity", "capacity_ladder"), ("rq4_ceilings", "cartpole_ceiling"),
                                                              ("simulator_validation", "emulator_reference"),
                                                              ("terminal_ablation", "capacity_ladder_no_terminal_adjustment")}
     or c == "pilot_60k" or v == "superseded_zero_init"),
    ("Output reuse (RQ3)", lambda c, s, v: c == "rq3_output_reuse" and s != "replicated_control"),
    ("Reuploading depth, with and without entanglement (RQ3)",
     lambda c, s, v: (c == "rq3_depth" and s in ("depth_sweep", "")) or (c == "rq3_entanglement" and s == "depth_sweep_unentangled")),
    ("FrozenLake (RQ4, RQ5)", lambda c, s, v: c == "rq5_frozenlake" or (c == "rq4_ceilings" and (s.startswith("frozenlake") or v == "cartpole_hparams")) or
     (c == "terminal_ablation" and s.startswith("frozenlake"))),
    ("Configurations of the reference study (RQ1)", lambda c, s, v: c == "rq1_porting"),
    ("Ablations, crossing and ceilings (RQ3, RQ4)", lambda c, s, v: (c == "rq3_depth" and s == "depth_reuse_crossing") or
     (c == "rq3_entanglement" and s == "reuse_sweep_unentangled")),
    ("Validation of the simulators", lambda c, s, v: c == "simulator_validation" and s != "emulator_reference" and v != "superseded_zero_init"),
    ("Additional results with 100 seeds", lambda c, s, v: c == "additional_x100"),
    ("Classical twin and the superseded control", lambda c, s, v: (c == "rq2_capacity" and s == "classical_twin") or
     (c == "rq3_output_reuse" and s == "replicated_control")),
    ("Learning-rate experiment", lambda c, s, v: c == "lr_experiment"),
]


def table_F1(R: Results) -> Table:
    runs = R.runs()
    runs["stage"] = runs["stage"].fillna("")
    runs["variant"] = runs["variant"].fillna("")
    runs["h"] = runs["wall_seconds"].astype(float) / 3600.0
    rows, tot = [], np.zeros(4)
    for label, pred in F1_ROWS:
        m = runs[[pred(c, s, v) for c, s, v in zip(runs.config, runs.stage, runs.variant)]]
        by = m.groupby("kind")["h"].sum()
        v = [by.get("real_circuit", 0.0), by.get("simulator", 0.0), by.get("classical", 0.0), float(len(m))]
        tot += v
        rows.append([label] + [f"{x:,.1f}" if x else "-" for x in v[:3]] + [f"{int(v[3]):,}"])
    rows.append(["Total"] + [f"{x:,.1f}" for x in tot[:3]] + [f"{int(tot[3]):,}"])
    return Table("F.1", "Wall-clock hours by experiment and type of arm, summed over runs",
                 ["Experiment", "Real circuit", "Simulators", "Classical", "Runs"], rows, "Table F.1")


def table_F2(R: Results) -> Table:
    runs = R.runs()
    runs["stage"] = runs["stage"].fillna("")
    items = [("CartPole", "Hybrid agent", (RQ2, "capacity_ladder", "hybrid")), ("CartPole", "Hybrid agent, R = 32", (RQ3, "reuse_sweep", "hybrid_reuse32")),
             ("CartPole", "Fourier ceiling", (RQ4, "cartpole_ceiling", "fourier_ceiling")), ("CartPole", "Large network", (RQ2, "capacity_ladder", "large_network")),
             ("FrozenLake", "Binary circuit, entangled, L = 5", (RQ5, "frozenlake_arms", "frozenlake_binary_depth5")),
             ("FrozenLake", "Scalar circuit, L = 15", (RQ5, "frozenlake_arms", "frozenlake_scalar_depth15")),
             ("FrozenLake", "Ceiling of the scalar circuit, L = 10", (RQ4, "frozenlake_ceilings", "frozenlake_scalar_ceiling_depth10")),
             ("FrozenLake", "Large network, one-hot input", (RQ5, "frozenlake_arms", "frozenlake_onehot_reference"))]
    rows = []
    for env, label, s in items:
        m = runs[(runs.config == s[0]) & (runs.stage == s[1]) & (runs.arm == s[2]) & (runs.variant.isna())]
        h = float(m.wall_seconds.astype(float).mean()) / 3600.0
        nd = 3 if env == "FrozenLake" else 1
        rows.append([env, label, n(num_params(s[2])), f"{h:.2f}" if h >= 0.005 else "<0.01", f"{iqm(vals(R, s, 'final')):.{nd}f}"])
    return Table("F.2", "Time per seed of representative arms, in hours of wall-clock time for 100k steps, with their parameter counts and training returns",
                 ["Environment", "Arm", "Par.", "Hours per seed", "Training return"], rows, "Table F.2")


def table_D2(R: Results) -> Table:
    items = [("Reference control", "reference_control"), ("Matched control", "matched_control"),
             ("Network without replication", "network_no_replication_w18"), ("Hybrid agent", "hybrid"),
             ("SU(2) emulator of the hybrid agent", "sim_emulator_hybrid_unentangled"), ("Classical twin", "classical_twin"),
             ("Fourier ceiling", "fourier_ceiling"), ("Fourier ceiling, distinct features", "fourier_ceiling_distinct"), ("Large network", "large_network")]
    rows = []
    for label, arm in items:
        total, split = _params_split(arm)
        rows.append([label, n(total), f"{split[0]} / {split[1]}" if split else "-"])
    return Table("D.2", "Structure of the CartPole arms, with measured parameter counts, in total and split between circuit or extractor and head",
                 ["Arm", "Par.", "Split"], rows, "Table D.2")


def _span(arms) -> str:
    p = [num_params(a) for a in arms]
    return f"{min(p):,}-{max(p):,}" if min(p) != max(p) else f"{p[0]:,}"


def table_33(R: Results) -> Table:
    """Parameter counts of the CartPole arms (table 3.3)."""
    rs = (4, 8, 16, 32)
    rows = [
        ["Hybrid agent, reuse R", _span(["hybrid"] + [f"hybrid_reuse{r}" for r in rs])],
        ["Reference control, reuse R", _span(["reference_control_reuse1", "reference_control"] + [f"reference_control_reuse{r}" for r in (8, 16, 32)])],
        ["Matched control, reuse R", _span(["matched_control"] + [f"matched_control_reuse{r}" for r in rs])],
        ["Network without replication", _span(["network_no_replication_w18"])],
        ["Large network", _span(["large_network"])],
        ["Classical twin, reuse R", _span(["classical_twin"] + [f"classical_twin_reuse{r}" for r in rs])],
        ["Hybrid agent, depth L", _span(["hybrid_depth1", "hybrid_depth2", "hybrid"])],
        ["Hybrid agent, depth L, reuse R", _span([f"hybrid_depth{L}_reuse{r}" for L in (1, 2) for r in rs])],
        ["Fourier ceiling", ", ".join(f"{num_params(a):,}" for a in ("fourier_ceiling", "fourier_ceiling_distinct"))]]
    return Table("3.3", "The arms used on CartPole, with measured parameter counts", ["Arm", "Par."], rows, "Table 3.3")


def table_34(R: Results) -> Table:
    """Parameter counts of the FrozenLake arms (table 3.4)."""
    rows = [
        ["Scalar circuit, depth L", _span([f"frozenlake_scalar_depth{L}" for L in (1, 5, 10, 15)])],
        ["Binary circuit, depth L", ", ".join(str(num_params(f"frozenlake_binary_depth{L}")) for L in (1, 5))],
        ["Unentangled binary circuit", ", ".join(str(num_params(f"frozenlake_binary_unentangled_depth{L}")) for L in (1, 5))],
        ["Matched control", _span(["frozenlake_matched_control"])],
        ["Large network, scalar input", _span(["frozenlake_large_network_scalar"])],
        ["Large network, one-hot input", _span(["frozenlake_onehot_reference"])],
        ["Network on the bits", ", ".join(str(num_params(f"frozenlake_bits_network_w{w}")) for w in (6, 12))],
        ["Fourier ceiling of the scalar circuit", _span([f"frozenlake_scalar_ceiling_depth{L}" for L in (1, 5, 10, 15)])],
        ["Linear ceiling of the binary circuit", _span(["frozenlake_binary_linear_ceiling"])]]
    return Table("3.4", "The arms used on FrozenLake, with measured parameter counts", ["Arm", "Par."], rows, "Table 3.4")


def _p_range(R: Results, pairs, metric: str, lr_stage: str = "dqn_twin") -> Tuple[float, float]:
    ps = [P(vals(R, (LR, lr_stage, a), metric), vals(R, (LR, lr_stage, b), metric)) for a, b in pairs]
    return min(ps), max(ps)


def table_43(R: Results) -> Table:
    """The predictions of the proposition in training (table 4.3): the quantities that the table and appendix E.1 quote."""
    tw, hy = "classical_twin", "sim_statevector_hybrid"
    rs = (4, 8, 16, 32)

    def reuse(prefix, r):
        return prefix if r == 1 else f"{prefix}_reuse{r}"

    rows = []
    for name, prefix, stage in (("Classical twin, DQN", tw, "dqn_twin"), ("Hybrid agent, simulator", hy, "dqn_simulator_hybrid")):
        col = [(f"{prefix}_collapsed{r}", reuse(prefix, r)) for r in rs]
        head = [(f"{prefix}_head_lr_x{r}", reuse(prefix, r)) for r in rs]
        for label, pairs in (("collapsed start vs reuse, R = 4..32", col), ("head x R from default initialisation vs reuse, R = 4..32", head)):
            lo_t, hi_t = _p_range(R, pairs, "final", stage)
            lo_g, hi_g = _p_range(R, pairs, "greedy", stage)
            rows.append([name, label, f"{lo_t:.2f}-{hi_t:.2f}", f"{lo_g:.2f}-{hi_g:.2f}"])
        grad = [(f"{prefix}_grad_x{r}", prefix) for r in (8, 32)]
        lo_t, hi_t = _p_range(R, grad, "final", stage)
        lo_g, hi_g = _p_range(R, grad, "greedy", stage)
        rows.append([name, "gradient x R (R = 8, 32) vs no reuse", f"{lo_t:.2f}-{hi_t:.2f}", f"{lo_g:.2f}-{hi_g:.2f}"])
        rows.append([name, "IQM of the gradient arms and of no reuse (training)",
                     " / ".join(f"{iqm(vals(R, (LR, stage, a), 'final')):.1f}" for a in (f"{prefix}_grad_x8", f"{prefix}_grad_x32", prefix)), ""])
    p128 = P(vals(R, (LR, "dqn_simulator_hybrid", f"{hy}_head_lr_x128"), "final"), vals(R, (LR, "dqn_simulator_hybrid", f"{hy}_reuse32"), "final"))
    rows.append(["Hybrid agent, simulator", "head x 128 vs reuse R = 32 (training)", f"{p128:.2f}", ""])
    # PPO: reuse against the multiplied learning rate, with and without clipping
    eq, inconclusive = 0, []
    for clip in ("clip", "noclip"):
        for r in rs:
            a, b = f"ppo_reference_lr_x{r}_{clip}", f"ppo_reference_reuse{r}_{clip}"
            pt = P(vals(R, (LR, "ppo", a), "final"), vals(R, (LR, "ppo", b), "final"))
            pg = P(vals(R, (LR, "ppo", a), "greedy"), vals(R, (LR, "ppo", b), "greedy"))
            ok = 0.35 <= pt <= 0.65 and 0.35 <= pg <= 0.65
            eq += int(ok)
            if not ok:
                inconclusive.append((clip, r, round(pt, 2), round(pg, 2)))
    # modified clipping rule: lr x R (modified clipping) against reuse with the ordinary clipping, R = 8 and 32
    for r in (8, 32):
        a, b = f"ppo_reference_lr_x{r}_modified_clip", f"ppo_reference_reuse{r}_clip"
        pt = P(vals(R, (LR, "ppo", a), "final"), vals(R, (LR, "ppo", b), "final"))
        pg = P(vals(R, (LR, "ppo", a), "greedy"), vals(R, (LR, "ppo", b), "greedy"))
        ok = 0.35 <= pt <= 0.65 and 0.35 <= pg <= 0.65
        eq += int(ok)
        if not ok:
            inconclusive.append(("modified clip", r, round(pt, 2), round(pg, 2)))
    rows.append(["Reference control, PPO", "lr x R vs reuse, with and without clipping and with the modified rule: equivalent comparisons out of 10", f"{eq} of 10",
                 "; ".join(f"R = {r} {c}: P = {pt:.2f}, {pg:.2f}" for c, r, pt, pg in inconclusive)])
    return Table("4.3", "The predictions of proposition 4.1 in training", ["Setting", "Comparison", "P training", "P greedy"], rows, "Table 4.3")


ALL.update({"C.1": table_C1, "E.5": table_E5, "4.1": table_41, "F.1": table_F1, "F.2": table_F2, "D.2": table_D2,
            "3.3": table_33, "3.4": table_34, "4.3": table_43})
