"""Tables of the supplementary results: validation of the simulators, additional contrasts with 100 seeds, cost, parameter counts
and the predictions of the proposition. The conventions are those of ``analysis``."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantumrl_dissection.analysis import (
    LR, REUSE, RQ1, RQ2, RQ3, RQ4, RQ5, P, Results, Table, _params_split, iqm, n, num_params, probability_of_improvement, vals,
)


def _boot_p(a, b, B: int = 10_000, seed: int = 0) -> Tuple[float, float]:
    """Percentile interval of P(a > b) under bootstrap resampling of the seeds of both arms."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ps = np.empty(B)
    for i in range(B):
        ps[i] = probability_of_improvement(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
    return float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))


def simulator_contrasts_table(R: Results) -> Table:
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
        pt, pg = P(da["training"], db["training"]), P(da["greedy"].dropna(), db["greedy"].dropna())
        lo, hi = _boot_p(da["training"], db["training"])
        rows.append([grp, label, f"{iqm(da['training']):.{nd}f}", f"{iqm(db['training']):.{nd}f}", f"{pt:.2f}", f"{lo:.2f}-{hi:.2f}", f"{pg:.2f}"])
    return Table("simulator_contrasts", "Additional contrasts computed with the batched state-vector simulator, 100 seeds per arm",
                 ["Group", "Contrast a > b", "IQM a", "IQM b", "P training", "95% interval", "P greedy"], rows)


def simulator_validation_table(R: Results) -> Table:
    _, v = evaluate_validation(R)
    rows = [["No global bias", f"d = {v['md']:+.3f} ({v['lo']:+.3f}, {v['hi']:+.3f}), {'met' if v['c1'] else 'not met'}"],
            ["No excess of disagreements", f"{v['outside']} of {v['cells']} outside, {'met' if v['c2'] else 'not met'}"],
            ["Agreement between simulators", f"{v['twin_bad']} of {v['twin_cells'] // 3} outside, {'met' if v['c3'] else 'not met'}"]]
    return Table("simulator_validation", "Outcome of the validation of the simulators against real arms with ten seeds each", ["Condition", "Outcome"], rows)


def reference_study_summary_table(R: Results) -> Table:
    from quantumrl_dissection.arms import REFERENCE_STUDY_RESULTS
    ppo = R.reference_summary()["mean"].to_dict()
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
        fin = [iqm(vals(R, (RQ1, "porting", a), "training")) for a in arms]
        ma = [float(np.mean(vals(R, (RQ1, "porting", a), "best_ma50"))) for a in arms]
        pp = [ppo[REFERENCE_STUDY_RESULTS[a]] for a in arms]
        pr = f"{min(par)}-{max(par)}" if min(par) != max(par) else str(par[0])
        rows.append([tmpl, sweep, pr, rng_s(fin, 1), rng_s(ma, 1), rng_s(pp, 1)])
    return Table("reference_study_summary", "The configurations of the reference study under DQN, by template, against their PPO counterparts",
                 ["Template", "Sweep", "Par.", "DQN training return", "DQN moving avg.", "PPO moving avg."], rows)


F1_ROWS = [
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


def cost_by_experiment_table(R: Results) -> Table:
    runs = R.runs()
    runs["stage"] = runs["stage"].fillna("")
    runs["variant"] = runs["variant"].fillna("")
    runs["h"] = runs["hours"]
    rows, tot = [], np.zeros(4)
    for label, pred in F1_ROWS:
        m = runs[[pred(c, s, v) for c, s, v in zip(runs.experiment, runs.stage, runs.variant)]]
        by = m.groupby("kind")["h"].sum()
        v = [by.get("real_circuit", 0.0), by.get("simulator", 0.0), by.get("classical", 0.0), float(len(m))]
        tot += v
        rows.append([label] + [f"{x:,.1f}" if x else "-" for x in v[:3]] + [f"{int(v[3]):,}"])
    rows.append(["Total"] + [f"{x:,.1f}" for x in tot[:3]] + [f"{int(tot[3]):,}"])
    return Table("cost_by_experiment", "Wall-clock hours by experiment and type of arm, summed over runs",
                 ["Experiment", "Real circuit", "Simulators", "Classical", "Runs"], rows)


def cost_per_seed_table(R: Results) -> Table:
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
        m = runs[(runs.experiment == s[0]) & (runs.stage == s[1]) & (runs.arm == s[2]) & (runs.variant.isna())]
        h = float(m.hours.mean())
        nd = 3 if env == "FrozenLake" else 1
        rows.append([env, label, n(num_params(s[2])), f"{h:.2f}" if h >= 0.005 else "<0.01", f"{iqm(vals(R, s, 'training')):.{nd}f}"])
    return Table("cost_per_seed", "Time per seed of representative arms, in hours of wall-clock time for 100k steps, with their parameter counts and training returns",
                 ["Environment", "Arm", "Par.", "Hours per seed", "Training return"], rows)


def cartpole_arm_structure_table(R: Results) -> Table:
    items = [("Reference control", "reference_control"), ("Matched control", "matched_control"),
             ("Network without replication", "network_no_replication_w18"), ("Hybrid agent", "hybrid"),
             ("SU(2) emulator of the hybrid agent", "sim_emulator_hybrid_unentangled"), ("Classical twin", "classical_twin"),
             ("Fourier ceiling", "fourier_ceiling"), ("Fourier ceiling, distinct features", "fourier_ceiling_distinct"), ("Large network", "large_network")]
    rows = []
    for label, arm in items:
        total, split = _params_split(arm)
        rows.append([label, n(total), f"{split[0]} / {split[1]}" if split else "-"])
    return Table("cartpole_arm_structure", "Structure of the CartPole arms, with measured parameter counts, in total and split between circuit or extractor and head",
                 ["Arm", "Par.", "Split"], rows)


def _span(arms) -> str:
    p = [num_params(a) for a in arms]
    return f"{min(p):,}-{max(p):,}" if min(p) != max(p) else f"{p[0]:,}"


def cartpole_arm_parameters_table(R: Results) -> Table:
    """Parameter counts of the CartPole arms."""
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
    return Table("cartpole_arm_parameters", "The arms used on CartPole, with measured parameter counts", ["Arm", "Par."], rows)


def frozenlake_arm_parameters_table(R: Results) -> Table:
    """Parameter counts of the FrozenLake arms."""
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
    return Table("frozenlake_arm_parameters", "The arms used on FrozenLake, with measured parameter counts", ["Arm", "Par."], rows)


def _p_range(R: Results, pairs, metric: str, lr_stage: str = "dqn_twin") -> Tuple[float, float]:
    ps = [P(vals(R, (LR, lr_stage, a), metric), vals(R, (LR, lr_stage, b), metric)) for a, b in pairs]
    return min(ps), max(ps)


def proposition_predictions_table(R: Results) -> Table:
    """The predictions of the proposition in training: the learning-rate variants against output reuse."""
    tw, hy = "classical_twin", "sim_statevector_hybrid"
    rs = (4, 8, 16, 32)

    def reuse(prefix, r):
        return prefix if r == 1 else f"{prefix}_reuse{r}"

    rows = []
    for name, prefix, stage in (("Classical twin, DQN", tw, "dqn_twin"), ("Hybrid agent, simulator", hy, "dqn_simulator_hybrid")):
        col = [(f"{prefix}_collapsed{r}", reuse(prefix, r)) for r in rs]
        head = [(f"{prefix}_head_lr_x{r}", reuse(prefix, r)) for r in rs]
        for label, pairs in (("collapsed start vs reuse, R = 4..32", col), ("head x R from default initialisation vs reuse, R = 4..32", head)):
            lo_t, hi_t = _p_range(R, pairs, "training", stage)
            lo_g, hi_g = _p_range(R, pairs, "greedy", stage)
            rows.append([name, label, f"{lo_t:.2f}-{hi_t:.2f}", f"{lo_g:.2f}-{hi_g:.2f}"])
        grad = [(f"{prefix}_grad_x{r}", prefix) for r in (8, 32)]
        lo_t, hi_t = _p_range(R, grad, "training", stage)
        lo_g, hi_g = _p_range(R, grad, "greedy", stage)
        rows.append([name, "gradient x R (R = 8, 32) vs no reuse", f"{lo_t:.2f}-{hi_t:.2f}", f"{lo_g:.2f}-{hi_g:.2f}"])
        rows.append([name, "IQM of the gradient arms and of no reuse (training)",
                     " / ".join(f"{iqm(vals(R, (LR, stage, a), 'training')):.1f}" for a in (f"{prefix}_grad_x8", f"{prefix}_grad_x32", prefix)), ""])
    p128 = P(vals(R, (LR, "dqn_simulator_hybrid", f"{hy}_head_lr_x128"), "training"), vals(R, (LR, "dqn_simulator_hybrid", f"{hy}_reuse32"), "training"))
    rows.append(["Hybrid agent, simulator", "head x 128 vs reuse R = 32 (training)", f"{p128:.2f}", ""])
    eq, inconclusive = 0, []
    for clip in ("clip", "noclip"):
        for r in rs:
            a, b = f"ppo_reference_lr_x{r}_{clip}", f"ppo_reference_reuse{r}_{clip}"
            pt = P(vals(R, (LR, "ppo", a), "training"), vals(R, (LR, "ppo", b), "training"))
            pg = P(vals(R, (LR, "ppo", a), "greedy"), vals(R, (LR, "ppo", b), "greedy"))
            ok = 0.35 <= pt <= 0.65 and 0.35 <= pg <= 0.65
            eq += int(ok)
            if not ok:
                inconclusive.append((clip, r, round(pt, 2), round(pg, 2)))
    for r in (8, 32):
        a, b = f"ppo_reference_lr_x{r}_modified_clip", f"ppo_reference_reuse{r}_clip"
        pt = P(vals(R, (LR, "ppo", a), "training"), vals(R, (LR, "ppo", b), "training"))
        pg = P(vals(R, (LR, "ppo", a), "greedy"), vals(R, (LR, "ppo", b), "greedy"))
        ok = 0.35 <= pt <= 0.65 and 0.35 <= pg <= 0.65
        eq += int(ok)
        if not ok:
            inconclusive.append(("modified clip", r, round(pt, 2), round(pg, 2)))
    rows.append(["Reference control, PPO", "lr x R vs reuse, with and without clipping and with the modified rule: equivalent comparisons out of 10", f"{eq} of 10",
                 "; ".join(f"R = {r} {c}: P = {pt:.2f}, {pg:.2f}" for c, r, pt, pg in inconclusive)])
    return Table("proposition_predictions", "The predictions of the proposition in training", ["Setting", "Comparison", "P training", "P greedy"], rows)


METRICS = ("training", "greedy", "auc")
CP, FL = "CartPole", "FrozenLake"
SIM = "simulator_validation"
Sel = Tuple[str, str, str]


def build_pairs() -> List[tuple]:
    pairs: List[tuple] = []

    def add(env, sim, real, glob=True, kind="sim-vs-real", max_sim_seed=None):
        pairs.append((env, sim, real, glob, kind, max_sim_seed))

    sv = lambda arm: (SIM, None, arm)                           # noqa: E731
    RC, RD, RE, R1, R5 = "rq3_output_reuse", "rq3_depth", "rq3_entanglement", "rq1_porting", "rq5_frozenlake"
    add(CP, sv("sim_statevector_hybrid"), ("rq2_capacity", "capacity_ladder", "hybrid"))
    add(CP, sv("sim_statevector_hybrid"), (RD, "depth_sweep", "hybrid"), glob=False)
    for r in (4, 8, 16, 32):
        add(CP, sv(f"sim_statevector_hybrid_reuse{r}"), (RC, "reuse_sweep", f"hybrid_reuse{r}"))
    for L in (1, 2):
        add(CP, sv(f"sim_statevector_hybrid_depth{L}"), (RD, "depth_sweep", f"hybrid_depth{L}"))
        for r in (4, 8, 16, 32):
            add(CP, sv(f"sim_statevector_hybrid_depth{L}_reuse{r}"), (RD, "depth_reuse_crossing", f"hybrid_depth{L}_reuse{r}"))
    for nq in (4, 8):
        for L in (1, 2, 5):
            add(CP, sv(f"sim_statevector_porting_skolik_{nq}q_depth{L}"), (R1, "porting", f"porting_skolik_{nq}q_depth{L}"))
    for L in (1, 2):
        add(CP, sv(f"sim_emulator_hybrid_unentangled_depth{L}"), (RE, "depth_sweep_unentangled", f"hybrid_unentangled_depth{L}"))
    emu = (SIM, "emulator_reference", "sim_emulator_hybrid_unentangled")
    add(CP, emu, (RE, "depth_sweep_unentangled", "hybrid_unentangled"))
    add(CP, emu, (RE, "reuse_sweep_unentangled", "hybrid_unentangled"), glob=False)
    for r in (4, 8, 16, 32):
        add(CP, sv(f"sim_emulator_hybrid_unentangled_reuse{r}"), (RE, "reuse_sweep_unentangled", f"hybrid_unentangled_reuse{r}"))
    for L in (2, 5):
        add(CP, sv(f"sim_emulator_porting_skolik_4q_depth{L}_unentangled"), (R1, "porting", f"porting_skolik_4q_depth{L}_unentangled"))
    for L in (1, 5):
        add(FL, sv(f"sim_statevector_frozenlake_binary_depth{L}"), (R5, "frozenlake_arms", f"frozenlake_binary_depth{L}"), max_sim_seed=10)
        add(FL, sv(f"sim_emulator_frozenlake_binary_unentangled_depth{L}"), (R5, "frozenlake_arms", f"frozenlake_binary_unentangled_depth{L}"))
    for L in (5, 10, 15):
        add(FL, sv(f"sim_emulator_frozenlake_scalar_depth{L}"), (R5, "frozenlake_arms", f"frozenlake_scalar_depth{L}"))
    add(FL, sv("sim_statevector_frozenlake_binary_depth5"), (R5, "frozenlake_arms", "frozenlake_binary_depth5"), glob=False)
    add(CP, sv("sim_statevector_hybrid_unentangled"), emu, kind="twin")
    for L in (1, 5):
        add(FL, sv(f"sim_statevector_frozenlake_binary_unentangled_depth{L}"), sv(f"sim_emulator_frozenlake_binary_unentangled_depth{L}"), kind="twin")
    add(CP, ("rq2_capacity", "capacity_ladder", "hybrid"), (RD, "depth_sweep", "hybrid"), glob=False, kind="real-vs-real")
    return pairs


def _vals(d: Dict[int, dict], k: str) -> list:
    return [r[k] for r in d.values() if r[k] is not None and not np.isnan(r[k])]


def _perm_p(rng, x, y, n=2000) -> float:
    obs = abs(probability_of_improvement(x, y) - 0.5)
    pool, k, hits = np.array(list(x) + list(y)), len(x), 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(probability_of_improvement(pool[:k], pool[k:]) - 0.5) >= obs - 1e-12:
            hits += 1
    return hits / n


def _std_diff(x, y) -> float:
    sd = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return (np.mean(x) - np.mean(y)) / sd if sd > 1e-9 else float("nan")


def evaluate_validation(R: Results):
    """(rows, verdict) of the validation, with the rule applied exactly as it was set."""
    load = _Loader(R)
    rng = np.random.default_rng(0)
    res, dlist, outside, cells, twin_bad, twin_cells, outside_cells = [], [], 0, 0, 0, 0, []
    for env, sim, real, glob, kind, max_sim in build_pairs():
        s, r = load(sim, max_sim), load(real)
        row = dict(kind=kind, env=env, sim=sim[2], real=real[2], n_sim=len(s), n_real=len(r), glob=glob)
        thr = 12.0 if env == CP else 0.05
        iq_s, iq_r = iqm(_vals(s, "training")), iqm(_vals(r, "training"))
        informative = (iq_s > thr) or (iq_r > thr)
        ps = {}
        for k in METRICS:
            x, y = _vals(s, k), _vals(r, k)
            ps[k] = probability_of_improvement(x, y) if x and y else float("nan")
        pp = _perm_p(rng, _vals(s, "training"), _vals(r, "training"))
        d = _std_diff(_vals(s, "training"), _vals(r, "training")) if informative else float("nan")
        in_rule = kind == "sim-vs-real" and glob and informative
        if in_rule:
            dlist.append(d)
            for k in METRICS:
                if not np.isnan(ps[k]):
                    cells += 1
                    bad = not (0.25 <= ps[k] <= 0.75)
                    outside += int(bad)
                    if bad:
                        outside_cells.append((sim[2], real[2], k, ps[k]))
        if kind == "twin":
            for k in METRICS:
                if not np.isnan(ps[k]):
                    twin_cells += 1
                    twin_bad += int(not (0.25 <= ps[k] <= 0.75))
        row.update(iq_s=iq_s, iq_r=iq_r, ps=ps, pp=pp, d=d, informative=informative, in_rule=in_rule)
        res.append(row)
    verdict = {}
    boot = [np.mean(rng.choice(dlist, len(dlist))) for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    md = float(np.mean(dlist))
    verdict.update(c1=(lo <= 0 <= hi) and abs(md) <= 0.30, md=md, lo=float(lo), hi=float(hi), n_inf=len(dlist),
                   c2=(outside / max(cells, 1)) <= 0.25, outside=outside, cells=cells,
                   c3=twin_cells > 0 and twin_bad == 0, twin_bad=twin_bad, twin_cells=twin_cells, outside_cells=outside_cells)
    verdict["accepted"] = bool(verdict["c1"] and verdict["c2"] and verdict["c3"])
    return res, verdict


class _Loader:
    """{seed: {training, greedy, auc}} of the runs of (experiment, stage, arm); a stage of None finds the one stage that holds the arm."""

    def __init__(self, R: Results):
        self.d = R._done[R._done.variant == ""]

    def __call__(self, sel, max_seed: Optional[int] = None) -> Dict[int, dict]:
        experiment, stage, arm = sel
        d = self.d[self.d.experiment == experiment]
        if stage is None:
            stages = set(d[d.arm == arm].stage)
            assert len(stages) == 1, (experiment, arm, stages)
            stage = stages.pop()
        d = d[(d.stage == stage) & (d.arm == arm)]
        if max_seed is not None:
            d = d[d.seed <= max_seed]
        return {int(r.seed): {"training": r.training, "greedy": None if pd.isna(r.greedy) else r.greedy, "auc": r.auc}
                for r in d.sort_values("seed").itertuples()}


TABLES: Dict[str, Callable[[Results], Table]] = {
    "simulator_validation": simulator_validation_table,
    "simulator_contrasts": simulator_contrasts_table,
    "reference_study_summary": reference_study_summary_table,
    "cost_by_experiment": cost_by_experiment_table,
    "cost_per_seed": cost_per_seed_table,
    "cartpole_arm_structure": cartpole_arm_structure_table,
    "cartpole_arm_parameters": cartpole_arm_parameters_table,
    "frozenlake_arm_parameters": frozenlake_arm_parameters_table,
    "proposition_predictions": proposition_predictions_table,
}
