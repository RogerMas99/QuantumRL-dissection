"""Loading, summarising and plotting run artefacts."""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

from quantumrl_dissection.training.provenance import is_simulated_manifest, warn_if_simulated


def load_episodes(csv_path: str | pathlib.Path):
    """Return (returns, global_steps) from an upstream episode CSV."""
    rewards, steps = [], []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            rewards.append(float(row["ep_reward"]))
            steps.append(int(row["global_step"]))
    return np.asarray(rewards), np.asarray(steps)


def load_eval(csv_path: str | pathlib.Path):
    steps, scores = [], []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["global_step"]))
            scores.append(float(row["greedy_return"]))
    return np.asarray(steps), np.asarray(scores)


def moving_average(x: np.ndarray, window: int = 50) -> np.ndarray:
    if len(x) < window:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode="valid")


def summarise_run(manifest: Dict[str, Any], tail_fraction: float = 0.2) -> Dict[str, Any]:
    """One row per run.

    `best_ma50` is the headline metric, not `last20`. DQN on CartPole decays
    from its peak even when healthy, so a tail statistic reports failure on runs
    that reached 200-400 mid-training.
    """
    out = manifest["outcome"]
    spec = manifest["spec"]
    rewards, steps = load_episodes(out["episodes_csv"])
    cut = steps >= (1 - tail_fraction) * out["total_timesteps"]

    row = {
        "arm": spec["arm"],
        "terminal_adjustment": spec["terminal_adjustment"],
        "seed": spec["seed"],
        "best_ma50": round(float(np.max(moving_average(rewards, 50))), 1),
        "tail_mean": round(float(rewards[cut].mean()), 1) if cut.any() else float("nan"),
        "mean_ep_len": round(float(rewards.mean()), 1),
        "n_episodes": len(rewards),
        "frac_poison": out["probe"]["frac_poison"],
        "wall_s": out["wall_seconds"],
        "simulated": is_simulated_manifest(manifest),
    }
    if out.get("eval_csv") and pathlib.Path(out["eval_csv"]).exists():
        _, scores = load_eval(out["eval_csv"])
        if len(scores):
            row["greedy_best"] = round(float(np.max(scores)), 1)
            row["greedy_final"] = round(float(scores[-1]), 1)
    return row


def collect(outdir: str | pathlib.Path) -> List[Dict[str, Any]]:
    """Every completed run under `outdir`, summarised."""
    rows = []
    for path in sorted(pathlib.Path(outdir).glob("*.manifest.json")):
        manifest = json.loads(path.read_text())
        if "error" in manifest:
            continue
        try:
            rows.append(summarise_run(manifest))
        except FileNotFoundError:
            continue
    return rows


def to_dataframe(outdir: str | pathlib.Path):
    import pandas as pd
    df = pd.DataFrame(collect(outdir))
    warn_if_simulated(df)      # policy: docs/reproducibility.md
    return df


def arm_comparison(outdir: str | pathlib.Path, metric: str = "best_ma50"):
    """Pivot: arms x terminal adjustment, mean and std across seeds."""
    import pandas as pd
    df = to_dataframe(outdir)
    if df.empty:
        return df
    return df.pivot_table(index="arm", columns="terminal_adjustment", values=metric,
                          aggfunc=["mean", "std", "count"]).round(1)


def plot_arms(outdir: str | pathlib.Path, arms: Optional[List[str]] = None,
              window: int = 50, savepath: Optional[str] = None):
    """Learning curves grouped by arm, coloured by the terminal-adjustment state."""
    import matplotlib.pyplot as plt

    manifests = [json.loads(p.read_text())
                 for p in sorted(pathlib.Path(outdir).glob("*.manifest.json"))]
    manifests = [m for m in manifests if "error" not in m]
    if arms is None:
        arms = sorted({m["spec"]["arm"] for m in manifests})

    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        for adjusted, colour in ((False, "tab:red"), (True, "tab:blue")):
            first = True
            for m in manifests:
                if m["spec"]["arm"] != arm or m["spec"]["terminal_adjustment"] != adjusted:
                    continue
                try:
                    rewards, steps = load_episodes(m["outcome"]["episodes_csv"])
                except FileNotFoundError:
                    continue
                ma = moving_average(rewards, window)
                ax.plot(steps[len(steps) - len(ma):], ma, color=colour,
                        alpha=0.65, lw=1.3,
                        label=("terminal adjustment on" if adjusted else "terminal adjustment off") if first else None)
                first = False
        ax.axhline(22, ls="--", c="gray", lw=0.8,
                   label="random policy (~22)" if arm == arms[0] else None)
        ax.set_title(arm)
        ax.set_xlabel("environment steps")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"return (MA-{window})")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
