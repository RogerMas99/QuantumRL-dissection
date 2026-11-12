"""Access to ``results/``: the runs table, the per-run learning curves and the per-seed metrics.

Everything is computed from the long-format files of ``results/<config>/`` (episodes, greedy evaluations). The metrics are the
ones of the study, with the definitions of ``analysis.stats``:

    final      mean return of the last 10 % of the training episodes (success rate on FrozenLake)
    greedy     mean of the last three greedy-evaluation checkpoints
    best_ma50  maximum of the 50-episode moving average of the training return (the statistic of the reference study)
    final_ma50 last value of that moving average
"""
from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from quantumrl_dissection.analysis import stats

DEFAULT_RESULTS = pathlib.Path(__file__).resolve().parents[3] / "results"


def _moving_average(x: np.ndarray, window: int = 50) -> np.ndarray:
    if len(x) < window:
        return x.copy()
    return np.convolve(x, np.ones(window) / window, mode="valid")


@lru_cache(maxsize=4)
def _runs(results: str) -> pd.DataFrame:
    return pd.read_csv(pathlib.Path(results) / "runs.csv", keep_default_na=False, na_values=[""])


def load_runs(results: pathlib.Path = DEFAULT_RESULTS) -> pd.DataFrame:
    """One row per run of the campaign (all of them, whether or not their curves are in ``results/``)."""
    return _runs(str(results)).copy()


def _config_files(results: pathlib.Path, config: str, name: str) -> pd.DataFrame:
    return pd.read_csv(pathlib.Path(results) / config / name, keep_default_na=False, na_values=[""])


def per_seed(config: str, results: pathlib.Path = DEFAULT_RESULTS) -> pd.DataFrame:
    """Per-seed metrics of every run of a configuration whose curves are in ``results/``."""
    results = pathlib.Path(results)
    ep = _config_files(results, config, "episodes.csv.gz")
    gr = _config_files(results, config, "greedy_eval.csv.gz")
    runs = load_runs(results)
    runs = runs[(runs.config == config) & (runs.in_results == True)]  # noqa: E712
    keys = ["stage", "arm", "variant", "seed"]
    ep["variant"] = ep["variant"].fillna("")
    gr["variant"] = gr["variant"].fillna("")
    runs = runs.copy()
    runs["variant"] = runs["variant"].fillna("")
    greedy = {k: g.sort_values("global_step")["greedy_return"].to_numpy(float) for k, g in gr.groupby(keys, sort=False)}
    rows = []
    for k, g in ep.groupby(keys, sort=False):
        g = g.sort_values("episode")
        rew = g["return"].to_numpy(float)
        ma = _moving_average(rew, 50)
        sc = greedy.get(k)
        rows.append(dict(zip(keys, k), final=stats.final_performance(rew, 0.1),
                         greedy=stats.greedy_final(sc) if sc is not None and len(sc) else np.nan,
                         best_ma50=float(np.nanmax(ma)), final_ma50=float(ma[-1]), n_episodes=len(rew)))
    out = pd.DataFrame(rows)
    out["seed"] = out["seed"].astype(int)
    out = out.merge(runs[keys + ["config", "steps", "wall_seconds", "terminal_adjustment", "terminal_adjustment_share", "environment",
                                 "kind", "algorithm"]], on=keys, how="left")
    return out.sort_values(keys).reset_index(drop=True)


@lru_cache(maxsize=8)
def _per_seed_cached(config: str, results: str) -> pd.DataFrame:
    return per_seed(config, pathlib.Path(results))


class Results:
    """Convenience view over ``results/``: select the per-seed metrics of an arm in a stage of a configuration."""

    def __init__(self, results: pathlib.Path = DEFAULT_RESULTS):
        self.results = pathlib.Path(results)

    def table(self, config: str) -> pd.DataFrame:
        return _per_seed_cached(config, str(self.results))

    def get(self, config: str, stage: str, arm: str, variant: str = "", include_reused: bool = False) -> pd.DataFrame:
        t = self.table(config)
        d = t[(t.stage == stage) & (t.arm == arm) & (t.variant.fillna("") == variant)]
        if len(d) == 0:
            raise KeyError(f"no runs for {config}/{stage}/{arm}")
        return d.sort_values("seed")

    def metric(self, config: str, stage: str, arm: str, metric: str, **kw) -> np.ndarray:
        return self.get(config, stage, arm, **kw)[metric].to_numpy(float)

    def runs(self) -> pd.DataFrame:
        """The runs table (``results/runs.csv``)."""
        return load_runs(self.results)

    def additional_x100(self, arm: str) -> pd.DataFrame:
        """The 100 seeds of an arm of the additional results: its own runs plus the runs it reuses from another configuration."""
        runs = load_runs(self.results)
        own = self.table("additional_x100")
        own = own[own.arm == arm]
        reused = runs[(runs.also_used_in == "additional_x100") & (runs.arm == arm)]
        parts = [own]
        for cfg, g in reused.groupby("config"):
            t = self.table(cfg)
            m = t.merge(g[["stage", "arm", "variant", "seed"]].assign(variant=lambda d: d["variant"].fillna("")),
                        on=["stage", "arm", "variant", "seed"])
            parts.append(m)
        return pd.concat(parts).sort_values("seed").reset_index(drop=True)
