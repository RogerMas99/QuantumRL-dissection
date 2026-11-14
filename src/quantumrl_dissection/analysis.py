"""Analysis: statistics, per-run metrics, the tables of the study and the validation of the simulators.

The statistics are the ones of the study. IQM is the mean of the values sorted and trimmed by floor(0.25 n) at each end; an
interval is the percentile 2.5 / 97.5 of 10,000 bootstrap resamples of the IQM (``default_rng(0)`` unless stated); P(a > b) is the
probability that a run of a beats a run of b, ties counting one half. A contrast reads "strong" when P >= 0.95 (or <= 0.05) in the
training and the greedy return, "level" when P is in [0.35, 0.65] in both, and "direction" otherwise.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

KEYS = ["stage", "arm", "variant", "seed"]


# --- statistics -----------------------------------------------------------------------------------------------------------------
def iqm(scores: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50% of the runs."""
    x = np.sort(np.asarray(scores, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if x.size < 4:
        return float(np.mean(x))
    lo, hi = int(np.floor(x.size * 0.25)), int(np.ceil(x.size * 0.75))
    return float(np.mean(x[lo:hi]))


def bootstrap_ci(scores: Sequence[float], statistic=iqm, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap interval of ``statistic``."""
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [statistic(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def probability_of_improvement(x: Sequence[float], y: Sequence[float]) -> float:
    """Probability that a run of ``x`` beats a run of ``y``, ties counted as half."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    b = np.asarray(y, dtype=float)
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    wins = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return float(wins / (a.size * b.size))


def final_performance(rewards: Sequence[float], last_frac: float = 0.1) -> float:
    """Mean return over the last ``last_frac`` of the episodes."""
    x = np.asarray(rewards, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(round(x.size * last_frac)))
    return float(np.mean(x[-k:]))


def greedy_final(scores: Sequence[float], last_n: int = 3) -> float:
    """Mean of the last ``last_n`` greedy-evaluation checkpoints."""
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x[-min(last_n, x.size):]))


def moving_average(x: np.ndarray, window: int = 50) -> np.ndarray:
    if len(x) < window:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode="valid")


# --- metrics of one run -------------------------------------------------------------------------------------------------------------
def run_metrics(episodes: pd.DataFrame, evaluations: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Metrics of one run from its episode returns (``ep_reward``, ``global_step``) and its greedy evaluations.

    training    mean return of the last 10 % of the episodes (success rate on FrozenLake)
    greedy      mean of the last three greedy-evaluation checkpoints
    best_ma50   maximum of the 50-episode moving average of the training return
    auc         area under that moving average, per step
    solved      the last greedy checkpoint reaches the goal on every evaluation episode
    """
    rew = episodes["ep_reward"].to_numpy(float)
    steps = episodes["global_step"].to_numpy()
    ma = moving_average(rew, 50)
    st = steps[len(rew) - len(ma):] if len(ma) < len(rew) else steps
    auc = float(np.trapezoid(ma, st) / max(st[-1] - st[0], 1)) if len(ma) > 1 else float("nan")
    scores = evaluations.sort_values("global_step")["greedy_return"].to_numpy(float) if evaluations is not None else np.array([])
    return {"training": final_performance(rew, 0.1), "greedy": greedy_final(scores) if len(scores) else np.nan,
            "best_ma50": float(np.nanmax(ma)), "auc": auc, "solved": bool(scores[-1] >= 0.999) if len(scores) else None}


# --- access to results/ -------------------------------------------------------------------------------------------------------------
class Results:
    """The files of ``results/``: the table of runs, the learning curves of the figures and the logs of the reference study."""

    def __init__(self, path="results"):
        self.path = pathlib.Path(path)
        read = lambda name: pd.read_csv(self.path / name, keep_default_na=False, na_values=[""])  # noqa: E731
        self._runs, self._curves, self._reference = read("runs.csv.gz"), read("curves.csv.gz"), read("reference_study.csv.gz")
        done = self._runs[self._runs.training.notna()].copy()
        done["variant"] = done["variant"].fillna("")
        self._done = done

    def runs(self) -> pd.DataFrame:
        """One row per run executed, those without curves in the figures and tables included."""
        return self._runs.copy()

    def get(self, experiment: str, stage: str, arm: str, variant: str = "") -> pd.DataFrame:
        d = self._done
        d = d[(d.experiment == experiment) & (d.stage == stage) & (d.arm == arm) & (d.variant == variant)]
        if len(d) == 0:
            raise KeyError(f"no runs for {experiment}/{stage}/{arm}")
        return d.sort_values("seed")

    def metric(self, experiment: str, stage: str, arm: str, metric: str, **kw) -> np.ndarray:
        return self.get(experiment, stage, arm, **kw)[metric].to_numpy(float)

    def additional_x100(self, arm: str) -> pd.DataFrame:
        """The 100 seeds of an arm of the additional results: its own runs plus the runs it shares with another experiment."""
        d = self._done
        d = d[((d.experiment == "additional_x100") | (d.also_used_in == "additional_x100")) & (d.arm == arm)]
        return d.sort_values("seed").reset_index(drop=True)

    def ma50_matrix(self, experiment: str, stage: str, arm: str) -> np.ndarray:
        """Seeds x steps matrix of the 50-episode moving average of the training return, every 1,000 steps."""
        c = self._curves
        c = c[(c.kind == "ma50") & (c.experiment == experiment) & (c.stage == stage) & (c.arm == arm)]
        return c.pivot(index="seed", columns="step", values="value").sort_index().sort_index(axis=1).to_numpy()

    def greedy_matrix(self, experiment: str, stage: str, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        """(steps, seeds x steps matrix) of the greedy evaluations."""
        c = self._curves
        c = c[(c.kind == "greedy") & (c.experiment == experiment) & (c.stage == stage) & (c.arm == arm)]
        m = c.pivot(index="seed", columns="step", values="value").sort_index().sort_index(axis=1)
        return m.columns.to_numpy(), m.to_numpy()

    def reference_logs(self, configs: Sequence[str], block: str = "DR") -> pd.DataFrame:
        r = self._reference
        return r[(r.record == "log") & (r.block == block) & r.config.isin(configs)]

    def reference_summary(self) -> pd.DataFrame:
        """Best 50-episode moving average of the reference study, mean and standard deviation over seeds, by configuration."""
        r = self._reference
        return r[r.record == "best_ma50"].set_index("config")[["value", "sd"]].rename(columns={"value": "mean"})


@dataclasses.dataclass
class Table:
    name: str
    title: str
    header: List[str]
    rows: List[List[str]]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.header)


def solved_last_checkpoint(R: Results, experiment: str, stage: str, arm: str) -> Tuple[int, int]:
    """Seeds whose last greedy checkpoint reaches the goal on every evaluation episode, and the number of seeds with evaluations."""
    d = R.runs()
    d = d[(d.experiment == experiment) & (d.stage == stage) & (d.arm == arm) & d.solved.notna()]
    return int(d.solved.astype(bool).sum()), int(len(d))


def interval(a, seed: int = 0) -> Tuple[float, float]:
    return bootstrap_ci(np.asarray(a, dtype=float), seed=seed)


def fmt_ci(a, nd: int = 1, seed: int = 0) -> str:
    lo, hi = interval(a, seed)
    return f"{iqm(a):.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def P(a, b) -> float:
    return probability_of_improvement(np.asarray(a, float), np.asarray(b, float))


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
    from quantumrl_dissection.models import build_agent_for, count_trainable
    from quantumrl_dissection.arms import build_arm_config, environment_of
    env = env_id or environment_of(arm)
    at, cfg = build_arm_config(arm, env_id=env)
    return count_trainable(build_agent_for(at, cfg, env_id=env, is_qnet=True))


def _params_split(arm: str) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Total parameters and, for the arms that have a circuit or an extractor, its split with the head."""
    from quantumrl_dissection.models import build_agent_for, count_trainable
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


S = Tuple[str, str, str]


def vals(R: Results, s: S, metric: str) -> np.ndarray:
    return R.metric(s[0], s[1], s[2], metric)


def cell(R: Results, s: S, nd: int = 1) -> Tuple[str, str]:
    """(training, greedy) cells 'IQM (lo-hi)', one bootstrap generator per statistic as in the tables of the study."""
    f, g = vals(R, s, "training"), vals(R, s, "greedy")
    return fmt_ci(f, nd), fmt_ci(g, nd)


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


def capacity_ladder_table(R: Results) -> Table:
    rows = []
    for label, s in CAPACITY:
        f, g = cell(R, s)
        rows.append([label, n(num_params(s[2])), f, g])
    return Table("capacity_ladder", "Results of the capacity ladder on CartPole: IQM and 95% interval over ten seeds",
                 ["Arm", "Par.", "Training return", "Greedy return"], rows)


def output_reuse_table(R: Results) -> Table:
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
    return Table("output_reuse", "Results of the output-reuse sweep on CartPole: IQM and 95% interval over ten seeds",
                 ["Arm", "R", "Par.", "Training return", "Greedy return"], rows)


def depth_table(R: Results) -> Table:
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
    return Table("depth", "Results of the reuploading depth sweep and of its crossing with output reuse on CartPole",
                 ["Arm", "L", "R", "Par.", "Training return", "Greedy return"], rows)


PORTING = [
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


def reference_study_configurations_table(R: Results) -> Table:
    rows = []
    for tmpl, q, L, r, ent, arm in PORTING:
        s = (RQ1, "porting", arm)
        f = vals(R, s, "training")
        g = vals(R, s, "greedy")
        rows.append([tmpl, str(q), str(L), r, ent, n(num_params(arm)), f"{iqm(f):.1f}", f"{iqm(g):.1f}"])
    return Table("reference_study_configurations", "The 24 configurations of the reference study run in RQ1, with measured parameter counts and IQM over ten seeds",
                 ["Template", "Qubits", "L", "R", "Ent.", "Par.", "Training return (IQM)", "Greedy return (IQM)"], rows)


def terminal_adjustment_table(R: Results) -> Table:
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
        on, off = vals(R, (RQ2 if env == "CartPole" else RQ5, stage_on, arm), "training"), vals(R, (TA, stage_off, arm + "_no_terminal_adjustment"), "training")
        ong = vals(R, (RQ2 if env == "CartPole" else RQ5, stage_on, arm), "greedy")
        offg = vals(R, (TA, stage_off, arm + "_no_terminal_adjustment"), "greedy")
        rows.append([env, label] + [f"{iqm(x):.{nd}f}" for x in (on, off, ong, offg)])
    return Table("terminal_adjustment", "Effect of the adjustment of terminal handling: IQM over ten seeds with and without it",
                 ["Environment", "Arm", "Training with", "Training without", "Greedy with", "Greedy without"], rows)


def hybrid_vs_controls_table(R: Results) -> Table:
    hy = (RQ2, "capacity_ladder", "hybrid")
    rows = []
    for label, s in [("Reference control", (RQ2, "capacity_ladder", "reference_control")),
                     ("Matched control", (RQ2, "capacity_ladder", "matched_control")),
                     ("Network without replication", (LR, "matched_control_with_reuse", "network_no_replication_w18")),
                     ("Classical twin, bounded", (RQ2, "classical_twin", "classical_twin")),
                     ("Classical twin, unbounded", (RQ2, "classical_twin", "classical_twin_unbounded"))]:
        pt, pg = P(vals(R, hy, "training"), vals(R, s, "training")), P(vals(R, hy, "greedy"), vals(R, s, "greedy"))
        rows.append([label, n(num_params(s[2])), f"{iqm(vals(R, s, 'training')):.1f}", f"{pt:.2f}", f"{pg:.2f}", reading(pt, pg)])
    return Table("hybrid_vs_controls", "The hybrid agent against the classical controls on CartPole",
                 ["Classical arm", "Par.", "Training return", "P training", "P greedy", "Reading"], rows)


def entanglement_table(R: Results) -> Table:
    rows = []
    items = [("Hybrid agent, L = 5, no reuse", (RQ2, "capacity_ladder", "hybrid"), (RQ3E, "reuse_sweep_unentangled", "hybrid_unentangled"))]
    for L in (1, 2, 5):
        items.append((f"Depth L = {L}", depth_selector(True, L), depth_selector(False, L)))
    for r in REUSE:
        items.append((f"Output reuse, R = {r}", reuse_selector("hybrid", r), reuse_selector("hybrid_unentangled", r)))
    for label, a, b in items:
        pt, pg = P(vals(R, a, "training"), vals(R, b, "training")), P(vals(R, a, "greedy"), vals(R, b, "greedy"))
        rows.append([label, f"{iqm(vals(R, a, 'training')):.1f}", f"{iqm(vals(R, b, 'training')):.1f}", f"{pt:.2f}", f"{pg:.2f}"])
    return Table("entanglement", "Entangled against unentangled circuits on CartPole",
                 ["Configuration", "Entangled", "Unentangled", "P training", "P greedy"], rows)


def depth_by_reuse_table(R: Results) -> Table:
    rows = []
    for L in (1, 2, 5):
        none = depth_selector(True, L)
        row = [f"L = {L}", f"{iqm(vals(R, none, 'training')):.1f}"]
        for r in REUSE:
            s = reuse_selector("hybrid", r) if L == 5 else (RQ3D, "depth_reuse_crossing", f"hybrid_depth{L}_reuse{r}")
            row.append(f"{iqm(vals(R, s, 'training')):.1f}")
        rows.append(row)
    rows[-1][1] = f"{iqm(vals(R, (RQ2, 'capacity_ladder', 'hybrid'), 'training')):.1f}"
    return Table("depth_by_reuse", "Depth crossed with output reuse on CartPole: IQM of the training return",
                 ["Depth", "No reuse", "R = 4", "R = 8", "R = 16", "R = 32"], rows)


def ceilings_table(R: Results) -> Table:
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
        pt, pg = P(vals(R, a, "training"), vals(R, b, "training")), P(vals(R, a, "greedy"), vals(R, b, "greedy"))
        rows.append([env, label, f"{iqm(vals(R, a, 'training')):.{nd}f}", f"{iqm(vals(R, b, 'training')):.{nd}f}", f"{pt:.2f}", f"{pg:.2f}",
                     reading(pt, pg)])
    return Table("ceilings", "The circuits against their classical ceilings",
                 ["Environment", "Contrast a > b", "IQM a", "IQM b", "P training", "P greedy", "Reading"], rows)


def frozenlake_arms_table(R: Results) -> Table:
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
        f, g = vals(R, s, "training"), vals(R, s, "greedy")
        lo, hi = interval(f)
        k, tot = solved_last_checkpoint(R, *s)
        rows.append([label, n(num_params(s[2])), f"{iqm(f):.3f}", f"{lo:.3f}-{hi:.3f}", f"{iqm(g):.3f}", f"{k}/{tot}"])
    return Table("frozenlake_arms", "All FrozenLake arms: success rate during training (IQM and 95% interval), greedy success rate (IQM), seeds solved",
                 ["Arm", "Par.", "Training IQM", "95% interval", "Greedy", "Seeds solved"], rows)


def _twin_arm(kind: str, k) -> str:
    return {"reuse": "classical_twin" if k == 1 else f"classical_twin_reuse{k}", "collapsed": f"classical_twin_collapsed{k}",
            "head": f"classical_twin_head_lr_x{k}", "grad": f"classical_twin_grad_x{k}", "all": f"classical_twin_all_lr_x{str(k).replace('.', 'p')}"}[kind]


def lr_dqn_table(R: Results) -> Table:
    rows = []
    layout = [("Reuse", "reuse", (1, 4, 8, 16, 32), "R = "), ("Collapsed start", "collapsed", (4, 8, 16, 32), "x"),
              ("Head", "head", (2, 4, 8, 16, 32, 64, 128), "x"), ("Gradient", "grad", (8, 32), "x"),
              ("All parameters", "all", (0.5, 2, 4, 8), "x")]
    for label, kind, ks, pre in layout:
        for k in ks:
            s = (LR, "dqn_twin", _twin_arm(kind, k))
            f, g = cell(R, s)
            rows.append([label, f"{pre}{k}", f, g])
    return Table("lr_dqn", "Learning rate in place of reuse on the classical twin under DQN: IQM and 95% interval over thirty seeds",
                 ["Setting", "Factor", "Training return", "Greedy return"], rows)


def lr_ppo_table(R: Results) -> Table:
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
    return Table("lr_ppo", "Learning rate in place of reuse under PPO: best 50-episode moving average, IQM and 95% interval over thirty seeds",
                 ["R", "Clip, reuse", "Clip, LR", "No clip, reuse", "No clip, LR"], rows)


def ceilings_and_networks_table(R: Results) -> Table:
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
        f, g = vals(R, s, "training"), vals(R, s, "greedy")
        rows.append([env, label, n(num_params(s[2])), fmt_ci(f, nd), fmt_ci(g, nd) if env == "CartPole" else f"{iqm(g):.3f}"])
    return Table("ceilings_and_networks", "The Fourier ceiling of CartPole on its distinct features, the network without replication and the networks on the bits",
                 ["Environment", "Arm", "Par.", "Training return", "Greedy return"], rows)


def best_moving_average_table(R: Results) -> Table:
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
        rows.append([env, label, f"{np.mean(b):.{nd}f} +- {np.std(b, ddof=1):.{nd}f}", f"{iqm(vals(R, s, 'training')):.{nd}f}",
                     f"{iqm(vals(R, s, 'greedy')):.{nd}f}"])
    return Table("best_moving_average", "The statistic of the reference study, the best 50-episode moving average (mean and standard deviation over seeds), "
                 "against the two statistics of this work (IQM)", ["Environment", "Arm", "Best moving average", "Training return", "Greedy return"],
                 rows)


TABLES: Dict[str, Callable[[Results], Table]] = {
    "terminal_adjustment": terminal_adjustment_table,
    "hybrid_vs_controls": hybrid_vs_controls_table,
    "entanglement": entanglement_table,
    "depth_by_reuse": depth_by_reuse_table,
    "ceilings": ceilings_table,
    "frozenlake_arms": frozenlake_arms_table,
    "capacity_ladder": capacity_ladder_table,
    "output_reuse": output_reuse_table,
    "depth": depth_table,
    "reference_study_configurations": reference_study_configurations_table,
    "lr_dqn": lr_dqn_table,
    "lr_ppo": lr_ppo_table,
    "ceilings_and_networks": ceilings_and_networks_table,
    "best_moving_average": best_moving_average_table,
}


def write_tables(R: Results, out: pathlib.Path, tables: Dict[str, Callable[[Results], Table]] = None,
                 only: Optional[Sequence[str]] = None) -> None:
    """Write each table as ``<name>.csv`` and ``<name>.md`` under ``out``."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (TABLES if tables is None else tables).items():
        if only and name not in only:
            continue
        t = fn(R)
        t.frame().to_csv(out / f"{name}.csv", index=False)
        md = [f"# {t.title}", "", "| " + " | ".join(t.header) + " |", "|" + "---|" * len(t.header)]
        md += ["| " + " | ".join(r) + " |" for r in t.rows]
        (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"{name}: {len(t.rows)} rows", flush=True)
