"""Validation of the simulators against the real circuit (appendix C.5, table C.1).

Acceptance rule, set before the validation runs finished:
 1. mean standardised difference d over the informative pairs: its 95 % bootstrap interval contains 0 and |mean d| <= 0.30
 2. at most 25 % of the (informative pair, metric) cells with P(simulator > real) outside [0.25, 0.75]
 3. the two simulators agree on the unentangled circuits: P in [0.25, 0.75] on every metric
Informative pair: IQM of the final return of at least one arm above 12 (CartPole) or 0.05 (FrozenLake). Pairs that reuse the same
simulator arm against a second real reference, and the real-against-real noise floor, are reported but are outside the global rule.

The simulator arms are trained on ten seeds and compared with the real arm, also of ten seeds, on three metrics: the final training
return, the greedy return and the area under the 50-episode moving average. The generator of the permutation test and of the
bootstrap of d is created once with seed 0 and consumed pair by pair in the order of ``build_pairs``.
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantumrl_dissection.analysis import stats
from quantumrl_dissection.analysis.data import Results, _moving_average

METRICS = ("final", "greedy", "auc")
CP, FL = "CartPole", "FrozenLake"
SIM = "simulator_validation"
Sel = Tuple[str, str, str]                       # (config, stage, arm)


def _all_metrics(R: Results, config: str) -> Dict[Tuple[str, str, int], dict]:
    """{(stage, arm, seed): {final, greedy, auc}} for the runs of a configuration, from the long-format curves."""
    ep = pd.read_csv(pathlib.Path(R.results) / config / "episodes.csv.gz", keep_default_na=False, na_values=[""])
    gr = pd.read_csv(pathlib.Path(R.results) / config / "greedy_eval.csv.gz", keep_default_na=False, na_values=[""])
    ep["variant"] = ep["variant"].fillna("")
    gr["variant"] = gr["variant"].fillna("")
    ep = ep[ep.variant == ""]
    gr = gr[gr.variant == ""]
    greedy = {k: g.sort_values("global_step")["greedy_return"].to_numpy(float) for k, g in gr.groupby(["stage", "arm", "seed"])}
    out = {}
    for k, g in ep.groupby(["stage", "arm", "seed"]):
        g = g.sort_values("episode")
        rew = g["return"].to_numpy(float)
        steps = g["global_step"].to_numpy()
        ma = _moving_average(rew, 50)
        st = steps[len(rew) - len(ma):] if len(ma) < len(rew) else steps
        auc = float(np.trapezoid(ma, st) / max(st[-1] - st[0], 1)) if len(ma) > 1 else float("nan")
        sc = greedy.get(k)
        out[k] = dict(final=stats.final_performance(rew, 0.1), greedy=stats.greedy_final(sc) if sc is not None and len(sc) else None, auc=auc)
    return out


class _Loader:
    def __init__(self, R: Results):
        self.R = R
        self._cache: Dict[str, dict] = {}

    def __call__(self, sel: Sel, max_seed: Optional[int] = None) -> Dict[int, dict]:
        config, stage, arm = sel
        if config not in self._cache:
            self._cache[config] = _all_metrics(self.R, config)
        m = self._cache[config]
        if stage is None:             # the stage that holds the arm (must be unique)
            stages = {k[0] for k in m if k[1] == arm}
            assert len(stages) == 1, (config, arm, stages)
            stage = stages.pop()
        return {k[2]: v for k, v in m.items() if k[0] == stage and k[1] == arm and (max_seed is None or k[2] <= max_seed)}


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
    obs = abs(stats.probability_of_improvement(x, y) - 0.5)
    pool, k, hits = np.array(list(x) + list(y)), len(x), 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(stats.probability_of_improvement(pool[:k], pool[k:]) - 0.5) >= obs - 1e-12:
            hits += 1
    return hits / n


def _std_diff(x, y) -> float:
    sd = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return (np.mean(x) - np.mean(y)) / sd if sd > 1e-9 else float("nan")


def evaluate(R: Results):
    """(rows, verdict) of the validation, with the rule applied exactly as it was set."""
    load = _Loader(R)
    rng = np.random.default_rng(0)
    res, dlist, outside, cells, twin_bad, twin_cells, outside_cells = [], [], 0, 0, 0, 0, []
    for env, sim, real, glob, kind, max_sim in build_pairs():
        s, r = load(sim, max_sim), load(real)
        row = dict(kind=kind, env=env, sim=sim[2], real=real[2], n_sim=len(s), n_real=len(r), glob=glob)
        thr = 12.0 if env == CP else 0.05
        iq_s, iq_r = stats.iqm(_vals(s, "final")), stats.iqm(_vals(r, "final"))
        informative = (iq_s > thr) or (iq_r > thr)
        ps = {}
        for k in METRICS:
            x, y = _vals(s, k), _vals(r, k)
            ps[k] = stats.probability_of_improvement(x, y) if x and y else float("nan")
        pp = _perm_p(rng, _vals(s, "final"), _vals(r, "final"))
        d = _std_diff(_vals(s, "final"), _vals(r, "final")) if informative else float("nan")
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
