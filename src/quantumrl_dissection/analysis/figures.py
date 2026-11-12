"""The result figures of the study (4.1 to 4.7), drawn from ``results/``.

    fig_capacity         4.1  capacity ladder on CartPole (per seed, IQM and 95% interval)
    fig_reuse            4.2  output reuse against the reuse factor R
    fig_reuse_curves     4.3  learning curves under output reuse (hybrid agent, classical twin)
    fig_lr               4.4  learning rate in place of output reuse (DQN twin, PPO reference control)
    fig_depth            4.5  reuploading depth, DQN against PPO
    fig_frozenlake       4.6  FrozenLake
    fig_key_contrasts    4.7  the key contrasts of each research question, as P(a > b)

The statistics are those of ``analysis.stats``; the generators of the bootstrap are seeded as stated in each function.
"""
from __future__ import annotations

import gzip
import pathlib
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quantumrl_dissection.analysis import stats  # noqa: E402
from quantumrl_dissection.analysis.data import Results  # noqa: E402
from quantumrl_dissection.analysis.tables import (  # noqa: E402
    LR, RQ1, RQ2, RQ3, RQ3D, RQ3E, RQ4, RQ5, P, depth_selector, iqm, reuse_selector, solved_last_checkpoint, vals,
)

BLUE, LIGHT_BLUE, GOLD, ORANGE, GREY, DARK, GREEN = "#1F3B8F", "#7FA3DC", "#C8961B", "#D2602A", "#8E8E8E", "#3A3A3A", "#2E7D4F"
RANDOM = 22.0


def _style(extra: Dict | None = None) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "axes.edgecolor": "#333333", "pdf.fonttype": 42, "axes.unicode_minus": False})
    if extra:
        plt.rcParams.update(extra)


def _iqm(a) -> float:
    a = np.sort(np.asarray(a, float))
    n = len(a)
    k = int(np.floor(n * 0.25))
    return a[k:n - k].mean()


def _ci(a, B: int = 10000, seed: int = 0):
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    s = np.array([_iqm(a[rng.integers(0, n, n)]) for _ in range(B)])
    return np.percentile(s, [2.5, 97.5])


def _random_band(ax, horizontal: bool = False) -> None:
    if horizontal:
        ax.axvspan(7, RANDOM, color="#F2F2F2", zorder=0, lw=0)
        ax.axvline(RANDOM, color="#9A9A9A", lw=0.7, ls=(0, (4, 3)), zorder=1.6)
    else:
        ax.axhspan(0, RANDOM, color="#F2F2F2", zorder=0, lw=0)
        ax.axhline(RANDOM, color="#9A9A9A", lw=0.7, ls=(0, (4, 3)), zorder=1.6)


# ---------------------------------------------------------------------------------------------------------------------
def fig_capacity(R: Results, out: str) -> None:
    """Figure 4.1: the capacity ladder (training and greedy return per seed)."""
    _style({"ytick.labelsize": 7.5})
    ARMS = [("Reference control (26)", (RQ2, "capacity_ladder", "reference_control"), "#8E8E8E"),
            ("Matched control (135)", (RQ2, "capacity_ladder", "matched_control"), "#D2602A"),
            ("Network without replication (128)", (LR, "matched_control_with_reuse", "network_no_replication_w18"), "#E39A6E"),
            ("Classical twin, bounded (132)", (RQ2, "classical_twin", "classical_twin"), "#C8961B"),
            ("Classical twin, unbounded (132)", (RQ2, "classical_twin", "classical_twin_unbounded"), "#8C6A12"),
            ("Hybrid agent (126)", (RQ2, "capacity_ladder", "hybrid"), "#1F3B8F"),
            ("Large network (10,934)", (RQ2, "capacity_ladder", "large_network"), "#3A3A3A")]
    fig, axs = plt.subplots(1, 2, figsize=(5.78, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(1)
    for j, (ax, title, met) in enumerate(zip(axs, ["(a) Training return", "(b) Greedy return"], ["final", "greedy"])):
        _random_band(ax, horizontal=True)
        for i, (lab, s, col) in enumerate(ARMS):
            v = vals(R, s, met)
            y = i + rng.uniform(-0.17, 0.17, len(v))
            ax.scatter(np.clip(v, 8, None), y, s=9, color=col, alpha=0.45, lw=0, zorder=2)
            lo, hi = _ci(v, seed=10 * i + j)
            ax.plot([lo, hi], [i, i], color=col, lw=2.6, solid_capstyle="round", zorder=3)
            ax.scatter([_iqm(v)], [i], s=46, color=col, edgecolor="white", lw=0.9, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(7, 560)
        ax.set_xticks([10, 22, 50, 100, 200, 500])
        ax.set_xticklabels(["10", "22", "50", "100", "200", "500"])
        ax.minorticks_off()
        ax.grid(axis="x", color="#E3E3E3", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("return (log scale)")
        ax.set_title(title, loc="left", pad=7)
        ax.text(RANDOM * 1.06, len(ARMS) - 0.3, "random policy", ha="left", va="center", fontsize=7.5, color="#8A8A8A")
        ax.tick_params(axis="y", length=0, labelcolor="#444444")
        ax.spines["left"].set_visible(False)
    axs[0].set_yticks(range(len(ARMS)))
    axs[0].set_yticklabels([a[0] for a in ARMS])
    axs[0].set_ylim(-0.6, len(ARMS) - 0.05)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_reuse(R: Results, out: str) -> None:
    """Figure 4.2: output reuse on CartPole, IQM and 95% interval against R."""
    _style({"ytick.labelsize": 8.5, "axes.edgecolor": "#333333"})
    RS = [1, 4, 8, 16, 32]
    fams = [("Hybrid agent, entangled", BLUE, "o", "hybrid"), ("Hybrid agent, unentangled", LIGHT_BLUE, "s", "hybrid_unentangled"),
            ("Classical twin", GOLD, "D", "twin"), ("Matched control", ORANGE, "^", "matched"), ("Reference control", GREY, "v", "reference")]

    def sel(key: str, r: int):
        if r == 1 and key == "hybrid":
            return (RQ2, "capacity_ladder", "hybrid")
        if r == 1 and key == "hybrid_unentangled":
            return (RQ3E, "reuse_sweep_unentangled", "hybrid_unentangled")
        if r == 1 and key == "matched":
            return (RQ2, "capacity_ladder", "matched_control")
        return reuse_selector(key, r)
    x = np.log2(RS)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.15), gridspec_kw={"wspace": 0.22})
    for ax, (met, title) in zip(axs, [("final", "(a) Training return"), ("greedy", "(b) Greedy return")]):
        _random_band(ax)
        ax.text(np.log2(1.6), RANDOM + 3, "random policy", fontsize=7.5, color="#8A8A8A", va="bottom")
        for lab, col, mk, key in fams:
            m, lo, hi = [], [], []
            for r in RS:
                v = vals(R, sel(key, r), met)
                m.append(_iqm(v))
                l, h = _ci(v)
                lo.append(l)
                hi.append(h)
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0, zorder=1)
            ax.plot(x, m, color=col, lw=1.6, marker=mk, ms=4.6, mec="white", mew=0.7, zorder=3, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(["1\n(none)", "4", "8", "16", "32"])
        ax.set_xlim(-0.3, 5.3)
        ax.set_ylim(0, None)
        ax.set_xlabel("reuse factor $R$")
        ax.set_title(title, loc="left", pad=4)
        ax.grid(axis="y", color="#E3E3E3", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("return (IQM, 95% CI)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=5, frameon=False, handlelength=1.8, columnspacing=1.1,
               handletextpad=0.4)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
def curves_ma50(R: Results, sel, grid: Sequence[int] = tuple(range(1000, 100001, 1000))) -> np.ndarray:
    """Seeds x grid matrix of the 50-episode moving average of the training return at each step of ``grid`` (values rounded to 3 decimals)."""
    config, stage, arm = sel
    ep = pd.read_csv(pathlib.Path(R.results) / config / "episodes.csv.gz", keep_default_na=False, na_values=[""])
    ep = ep[(ep.stage == stage) & (ep.arm == arm)]
    rows = []
    for seed, g in ep.groupby("seed"):
        g = g.sort_values("episode")
        rew = g["return"].to_numpy(float)
        steps = g["global_step"].to_numpy()
        ma = np.array([rew[max(0, i - 49): i + 1].mean() for i in range(len(rew))])
        row = []
        for edge in grid:
            idx = np.searchsorted(steps, edge, side="right") - 1
            row.append(round(float(ma[idx]), 3) if idx >= 0 else np.nan)
        rows.append(row)
    return np.array(rows)


def _iqm_rows(M: np.ndarray) -> np.ndarray:
    S = np.sort(M, axis=0)
    n = S.shape[0]
    a = int(np.floor(n * 0.25))
    return S[a:n - a].mean(axis=0)


def _curve(M: np.ndarray, B: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    bs = np.array([_iqm_rows(M[rng.integers(0, n, n)]) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return _iqm_rows(M), lo, hi


def fig_reuse_curves(R: Results, out: str) -> None:
    """Figure 4.3: learning curves under output reuse."""
    _style({"legend.fontsize": 8})
    grid = np.arange(1000, 100001, 1000)
    PANELS = [("(a) Hybrid agent",
               [((RQ2, "capacity_ladder", "hybrid"), "no reuse", "#A9BCE3", 1.4), (reuse_selector("hybrid", 4), "$R = 4$", "#6F8FD0", 1.4),
                (reuse_selector("hybrid", 16), "$R = 16$", "#3A5DB0", 1.4), (reuse_selector("hybrid", 32), "$R = 32$", "#1F3B8F", 1.8)]),
              ("(b) Classical twin",
               [(reuse_selector("twin", 1), "no reuse", "#EBD39A", 1.4), (reuse_selector("twin", 4), "$R = 4$", "#DDB45A", 1.4),
                (reuse_selector("twin", 16), "$R = 16$", "#C8961B", 1.4), (reuse_selector("twin", 32), "$R = 32$", "#8C6A12", 1.8)])]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.32), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (title, arms) in zip(axs, PANELS):
        _random_band(ax)
        for sel, lab, col, lw in arms:
            m, lo, hi = _curve(curves_ma50(R, sel))
            ax.fill_between(grid / 1e3, lo, hi, color=col, alpha=0.15, lw=0, zorder=1)
            ax.plot(grid / 1e3, m, color=col, lw=lw, label=lab, zorder=3)
        ax.set_title(title, loc="left", pad=5)
        ax.set_xlim(0, 100)
        ax.set_xlabel(r"environment steps ($\times 10^3$)")
        ax.grid(axis="y", color="#E3E3E3", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    axs[0].text(99, RANDOM - 2.5, "random policy", fontsize=7, color="#8A8A8A", ha="right", va="top", zorder=1.7)
    axs[0].set_ylabel("training return (50-episode mean)")
    axs[0].set_ylim(0, None)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_lr(R: Results, out: str) -> None:
    """Figure 4.4: learning rate in place of output reuse (DQN classical twin; PPO reference control)."""
    _style()
    tw = lambda arm, col="final": vals(R, (LR, "dqn_twin", arm), col)       # noqa: E731
    pp = lambda arm: vals(R, (LR, "ppo", arm), "best_ma50")                  # noqa: E731
    DX = 0.13
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.15, 2.45), gridspec_kw={"wspace": 0.25})
    K = [1, 2, 4, 8, 16, 32, 64, 128]
    arms = ["classical_twin"] + [f"classical_twin_head_lr_x{k}" for k in K[1:]]
    m = [_iqm(tw(x)) for x in arms]
    lo, hi = np.array([_ci(tw(x)) for x in arms]).T
    xk = np.log2(K)
    a.axhspan(0, RANDOM, color="#F2F2F2", zorder=0, lw=0)
    a.axhline(RANDOM, color="#9A9A9A", lw=0.7, ls=(0, (4, 3)), zorder=1.6)
    a.text(7.3, RANDOM - 3, "random policy", fontsize=7.5, color="#8A8A8A", ha="right", va="top", zorder=1.7)
    a.fill_between(xk, lo, hi, color=BLUE, alpha=0.13, lw=0, zorder=1)
    a.plot(xk, m, color=BLUE, lw=1.6, marker="o", ms=3.2, mec="white", mew=0.6, zorder=3, label=r"learning rate of the first layer $\times k$")
    Rs = [1, 4, 8, 16, 32]
    rm = [_iqm(tw("classical_twin" if r == 1 else f"classical_twin_reuse{r}")) for r in Rs]
    a.plot(np.log2(Rs) + np.r_[0, [DX] * 4], rm, ls="none", marker="D", ms=2.8, color=GOLD, mec=GOLD, zorder=4, label=r"output reuse $R$")
    cm = [_iqm(tw(f"classical_twin_collapsed{r}")) for r in Rs[1:]]
    a.plot(np.log2(Rs[1:]) - DX, cm, ls="none", marker="D", ms=3.8, mfc="white", mec=GOLD, mew=1.0, zorder=4, label=r"learning rate $\times R$, collapsed start")
    gm = [_iqm(tw(f"classical_twin_grad_x{r}")) for r in (8, 32)]
    a.plot(np.log2([8, 32]), gm, ls="none", marker="s", ms=3.0, color=GREY, zorder=4, label=r"gradient $\times R$")
    a.set_xticks(xk)
    a.set_xticklabels([str(k) for k in K])
    a.set_xlim(-0.4, 7.4)
    a.set_ylim(0, None)
    a.set_xlabel(r"$R$ or $k$")
    a.set_ylabel("training return (IQM, 95% CI)")
    a.set_title("(a) DQN, classical twin", loc="left", pad=4)
    a.grid(axis="y", color="#E3E3E3", lw=0.6, zorder=0)
    a.set_axisbelow(True)
    a.legend(loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
    pm = [_iqm(pp(f"ppo_reference_reuse{r}_clip")) for r in Rs]
    plo, phi = np.array([_ci(pp(f"ppo_reference_reuse{r}_clip")) for r in Rs]).T
    xr = np.log2(Rs)
    b.axhspan(0, RANDOM, color="#F2F2F2", zorder=0, lw=0)
    b.axhline(RANDOM, color="#9A9A9A", lw=0.7, ls=(0, (4, 3)), zorder=1.6)
    b.text(0.25, RANDOM + 5, "random policy", fontsize=7.5, color="#8A8A8A", va="bottom", zorder=1.7)
    b.fill_between(xr, plo, phi, color=GOLD, alpha=0.15, lw=0, zorder=1)
    b.plot(xr, pm, color=GOLD, lw=1.6, marker="D", ms=2.6, zorder=3, label=r"output reuse $R$")
    lm = [_iqm(pp(f"ppo_reference_lr_x{r}_clip")) for r in Rs[1:]]
    b.plot(np.log2(Rs[1:]) + DX, lm, color=BLUE, lw=1.3, ls=(0, (4, 1.5)), marker="o", ms=3.8, mfc="white", mec=BLUE, mew=1.0, zorder=4,
           label=r"learning rate $\times R$")
    # the logs of the reference study at R = 4, 8, 16, 32 (data/reference_study)
    summ = pd.read_csv(pathlib.Path(R.results) / "reference_study" / "reference_study_ppo_summary.csv").set_index("config")
    ref_m = [float(summ.loc[f"Classical_r{r}", "best_ma50_mean"]) for r in Rs[1:]]
    ref_s = [float(summ.loc[f"Classical_r{r}", "best_ma50_sd"]) for r in Rs[1:]]
    b.errorbar(np.log2(Rs[1:]) - DX, ref_m, yerr=ref_s, ls="none", marker="s", ms=4, color="#333333", capsize=2, lw=0.9, zorder=5,
               label=r"reference study (mean $\pm$ sd)")
    b.set_xticks(xr)
    b.set_xticklabels([str(r) for r in Rs])
    b.set_xlim(-0.4, 5.4)
    b.set_ylim(0, None)
    b.set_xlabel(r"$R$")
    b.set_ylabel("best 50-episode moving average")
    b.set_title("(b) PPO, reference control", loc="left", pad=4)
    b.grid(axis="y", color="#E3E3E3", lw=0.6, zorder=0)
    b.set_axisbelow(True)
    b.legend(loc="lower right", bbox_to_anchor=(1, 0.12), frameon=False, handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
def _paper_ma50(R: Results, configs: Sequence[str], grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Moving average (50 episodes) of the per-seed logs of the reference study, on ``grid`` (same rule as ``curves_ma50``)."""
    path = pathlib.Path(R.results) / "reference_study" / "reference_study_ppo_curves.csv.gz"
    df = pd.read_csv(path)
    df = df[(df.block == "DR") & (df.config.isin(configs))]
    out: Dict[str, List[np.ndarray]] = {c: [] for c in configs}
    for (c, seed), g in df.groupby(["config", "seed"]):
        g = g.sort_values("step")
        st, ret = g["step"].to_numpy(), g["ep_return"].to_numpy(float)
        ma = np.array([ret[max(0, i - 49): i + 1].mean() for i in range(len(ret))])
        row = []
        for edge in grid:
            idx = np.searchsorted(st, edge, side="right") - 1
            row.append(ma[idx] if idx >= 0 else np.nan)
        out[c].append(np.array(row))
    return {c: np.array(v) for c, v in out.items()}


def fig_depth(R: Results, out: str) -> None:
    """Figure 4.5: reuploading depth on the eight-qubit Skolik template, DQN (this work) and PPO (reference study)."""
    _style({"legend.fontsize": 7.5})
    grid = np.arange(1000, 100001, 1000)
    cols = {1: "#B9CBE6", 2: "#5F86C2", 5: "#1F3B8F"}
    paper = _paper_ma50(R, [f"Skolik_8Q_L{L}" for L in (1, 2, 5)], grid)
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.32), sharey=True, gridspec_kw={"wspace": 0.09})
    for L in (1, 2, 5):
        M = curves_ma50(R, depth_selector(True, L), grid)
        for ax, MM in ((axs[0], M), (axs[1], paper[f"Skolik_8Q_L{L}"])):
            m, lo, hi = _curve(MM, B=4000)
            ax.fill_between(grid / 1e3, lo, hi, color=cols[L], alpha=0.15, lw=0, zorder=1)
            ax.plot(grid / 1e3, m, color=cols[L], lw=1.6, label=f"$L = {L}$", zorder=3)
    for ax, title in zip(axs, ["(a) DQN, this work", "(b) PPO, reference study"]):
        _random_band(ax)
        ax.set_title(title, loc="left", pad=5)
        ax.set_xlim(0, 100)
        ax.set_xlabel(r"environment steps ($\times 10^3$)")
        ax.grid(axis="y", color="#E3E3E3", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axs[1].spines["left"].set_visible(False)
    axs[1].tick_params(axis="y", length=0)
    axs[0].legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    axs[0].set_ylabel("training return (50-episode mean)")
    axs[0].set_ylim(0, None)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _greedy_matrix(R: Results, config: str, stage: str, arm: str) -> Tuple[np.ndarray, np.ndarray]:
    gr = pd.read_csv(pathlib.Path(R.results) / config / "greedy_eval.csv.gz", keep_default_na=False, na_values=[""])
    gr = gr[(gr.stage == stage) & (gr.arm == arm)]
    steps = np.sort(gr.global_step.unique())
    M = np.array([[g.set_index("global_step").greedy_return.get(s, np.nan) for s in steps] for _, g in gr.groupby("seed")])
    return steps, M


def fig_frozenlake(R: Results, out: str) -> None:
    """Figure 4.6: FrozenLake. (a) seeds whose greedy policy reaches the goal at each checkpoint; (b) scalar circuit against its ceilings."""
    _style({"legend.fontsize": 7})
    fig, axs = plt.subplots(1, 2, figsize=(7.15, 2.7), gridspec_kw={"wspace": 0.3})
    ax = axs[0]
    for lab, sel, col, mk, ls in [("One-hot network (5,508)", (RQ5, "frozenlake_arms", "frozenlake_onehot_reference"), DARK, "D", "-"),
                                  ("Binary circuit, entangled (60)", (RQ5, "frozenlake_arms", "frozenlake_binary_depth5"), BLUE, "o", "-"),
                                  ("Binary circuit, unentangled (60)", (RQ5, "frozenlake_arms", "frozenlake_binary_unentangled_depth5"), LIGHT_BLUE, "s", "-"),
                                  ("Linear ceiling (20)", (RQ4, "frozenlake_ceilings", "frozenlake_binary_linear_ceiling"), GREEN, "^", "-")]:
        steps, M = _greedy_matrix(R, *sel)
        ax.plot(steps / 1e3, (M >= 0.999).mean(axis=0), marker=mk, ms=3.2, lw=1.6, color=col, label=lab, ls=ls, mec="white", mew=0.4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(5, 105)
    ax.set_xlabel(r"environment steps ($\times 10^3$)")
    ax.set_ylabel("fraction of seeds solved (greedy)")
    ax.set_title("(a) Binary circuit, $L = 5$", loc="left", pad=5)
    ax.grid(axis="y", color="#E3E3E3", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2, handlelength=1.8, columnspacing=1.0)
    ax = axs[1]
    Ls = [1, 5, 10, 15]
    for lab, key, col, mk, off in [("Scalar circuit", lambda L: (RQ5, "frozenlake_arms", f"frozenlake_scalar_depth{L}"), LIGHT_BLUE, "s", -0.0),
                                   ("Fourier ceiling", lambda L: (RQ4, "frozenlake_ceilings", f"frozenlake_scalar_ceiling_depth{L}"), GREEN, "^", 0.0)]:
        m, lo, hi = [], [], []
        for L in Ls:
            v = vals(R, key(L), "final")
            m.append(_iqm(v))
            l, h = _ci(v)
            lo.append(l)
            hi.append(h)
        x = np.arange(4)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0, zorder=1)
        ax.plot(x, m, color=col, marker=mk, ms=3.8, lw=1.6, mec="white", mew=0.5, label=lab, zorder=3)
    onehot = _iqm(vals(R, (RQ5, "frozenlake_arms", "frozenlake_onehot_reference"), "final"))
    ax.axhline(onehot, color="#222222", ls=(0, (4, 3)), lw=0.9)
    ax.text(3.1, onehot - 0.04, "one-hot network", fontsize=7, ha="right", va="top", color="#333333")
    ax.set_xticks(range(4))
    ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("depth $L$")
    ax.set_ylabel("success rate (training)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("(b) Scalar circuit and its ceilings", loc="left", pad=5)
    ax.grid(axis="y", color="#E3E3E3", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2, handlelength=1.8)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
def key_contrasts(R: Results) -> List[Tuple[str, str, float, float]]:
    """(research question, label, P training, P greedy) of the contrasts that figure 4.7 places on one scale."""
    ent = (RQ2, "capacity_ladder", "hybrid")
    unent = (RQ3E, "reuse_sweep_unentangled", "hybrid_unentangled")
    tw16 = reuse_selector("twin", 16)
    tw1 = reuse_selector("twin", 1)
    bin5 = (RQ5, "frozenlake_arms", "frozenlake_binary_depth5")
    items = [
        ("RQ2", "Hybrid agent $>$ reference control", ent, (RQ2, "capacity_ladder", "reference_control")),
        ("RQ2", "Hybrid agent $>$ matched control", ent, (RQ2, "capacity_ladder", "matched_control")),
        ("RQ2", "Hybrid agent $>$ network without replication", ent, (LR, "matched_control_with_reuse", "network_no_replication_w18")),
        ("RQ2", "Hybrid agent $>$ classical twin", ent, (RQ2, "classical_twin", "classical_twin_unbounded")),
        ("RQ3", "Hybrid agent, $R = 32$ $>$ no reuse", reuse_selector("hybrid", 32), ent),
        ("RQ3", "Classical twin, $R = 16$ $>$ no reuse", tw16, tw1),
        ("RQ3", "PPO reference control, $R = 4$ $>$ $R = 1$", (LR, "ppo", "ppo_reference_reuse4_clip"), (LR, "ppo", "ppo_reference_reuse1_clip")),
        ("RQ3", "Depth $L = 2$ $>$ $L = 1$", depth_selector(True, 2), depth_selector(True, 1)),
        ("RQ3", "Depth $L = 5$ $>$ $L = 2$", depth_selector(True, 5), depth_selector(True, 2)),
        ("RQ3", "Entangled $>$ unentangled, no reuse", ent, unent),
        ("RQ3", "Entangled $>$ unentangled, $R = 32$", reuse_selector("hybrid", 32), reuse_selector("hybrid_unentangled", 32)),
        ("RQ4", "CartPole ceiling $>$ unentangled circuit", (LR, "ceiling_with_reuse", "fourier_ceiling_distinct"), unent),
        ("RQ4", "Scalar ceiling $>$ scalar circuit, $L = 10$", (RQ4, "frozenlake_ceilings", "frozenlake_scalar_ceiling_depth10"),
         (RQ5, "frozenlake_arms", "frozenlake_scalar_depth10")),
        ("RQ4", "Binary entangled $>$ unentangled", bin5, (RQ5, "frozenlake_arms", "frozenlake_binary_unentangled_depth5")),
        ("RQ4", "Binary entangled $>$ linear ceiling", bin5, (RQ4, "frozenlake_ceilings", "frozenlake_binary_linear_ceiling")),
        ("RQ5", "Binary circuit, $L = 5$ $>$ $L = 1$", bin5, (RQ5, "frozenlake_arms", "frozenlake_binary_depth1")),
        ("RQ5", "Binary circuit $>$ network on bits (58)", bin5, (LR, "bits_networks", "frozenlake_bits_network_w6")),
        ("RQ5", "Binary circuit $>$ one-hot network (5,508)", bin5, (RQ5, "frozenlake_arms", "frozenlake_onehot_reference")),
    ]
    out = []
    for rq, label, a, b in items:
        out.append((rq, label, P(vals(R, a, "final"), vals(R, b, "final")), P(vals(R, a, "greedy"), vals(R, b, "greedy"))))
    return out


def fig_key_contrasts(R: Results, out: str) -> None:
    """Figure 4.7: the key contrasts of each research question as P(a > b), in the training return (circle) and the greedy return (diamond)."""
    _style({"ytick.labelsize": 7.5, "xtick.labelsize": 7.5})
    rows = key_contrasts(R)
    fig, ax = plt.subplots(figsize=(5.9, 4.6))
    ax.axvspan(0, 0.05, color="#E3ECF7", lw=0, zorder=0)
    ax.axvspan(0.95, 1.0, color="#E3ECF7", lw=0, zorder=0)
    ax.axvspan(0.35, 0.65, color="#F2F2F2", lw=0, zorder=0)
    ax.axvline(0.5, color="#9A9A9A", lw=0.7, ls=(0, (4, 3)), zorder=1)
    y, ys, labels, gaps = 0.0, [], [], {}
    prev = None
    for rq, label, pt, pg in rows:
        if prev is not None and rq != prev:
            y += 0.9
        gaps.setdefault(rq, []).append(y)
        ys.append(y)
        labels.append(label)
        ax.plot([min(pt, pg), max(pt, pg)], [y, y], color="#C8C8C8", lw=1.0, zorder=2)
        ax.scatter([pt], [y], s=14, color=BLUE, zorder=3)
        ax.scatter([pg], [y], s=22, marker="D", facecolor="white", edgecolor=BLUE, lw=1.0, zorder=3)
        y += 1.0
        prev = rq
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab in labels], fontsize=7)
    ax.invert_yaxis()
    for rq, ypos in gaps.items():
        ax.text(-0.57, float(np.mean(ypos)), rq, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5, color="#444444")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.05, 0.35, 0.5, 0.65, 0.95, 1.0])
    ax.set_xticklabels(["0", ".05", ".35", ".5", ".65", ".95", "1"])
    ax.set_xlabel(r"probability that the first arm scores higher, $P(a > b)$")
    ax.text(0.5, 1.005, "level", ha="center", va="bottom", fontsize=6.5, color="#777777", transform=ax.get_xaxis_transform())
    ax.text(0.975, 1.005, "strong", ha="center", va="bottom", fontsize=6.5, color="#5D7DB5", transform=ax.get_xaxis_transform())
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="none", color=BLUE, ms=4, label="training return"),
         plt.Line2D([], [], marker="D", ls="none", mfc="white", mec=BLUE, ms=4.5, label="greedy return")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=7.5)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


FIGURES: Dict[str, Callable[[Results, str], None]] = {
    "4.1": fig_capacity, "4.2": fig_reuse, "4.3": fig_reuse_curves, "4.4": fig_lr, "4.5": fig_depth, "4.6": fig_frozenlake, "4.7": fig_key_contrasts,
}
