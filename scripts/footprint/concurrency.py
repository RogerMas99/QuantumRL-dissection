"""How many runs shared the local machine at the same time? (decides the single/parallel calibration condition)

For every run recorded as made on the local machine (``log_origin == "local"`` in results/runs.csv) the interval is
[started_utc, started_utc + wall_seconds]. A sweep over all intervals gives, per run, the time-weighted mean number of runs alive
during it (itself included). Only runs listed in runs.csv are counted (not tests, traces or analyses).

    python scripts/footprint/concurrency.py --runs results/runs.csv --out docs/footprint/local_concurrency.csv
"""
import argparse

import numpy as np
import pandas as pd


def kind_of_run(r) -> str:
    """Calibration kind of a row of runs.csv."""
    if r.algorithm == "ppo":
        return "ppo"
    if r.kind == "real_circuit":
        return "real_frozen" if r.environment == "FrozenLake" else "real_cartpole"
    if r.kind == "simulator":
        return "sv_cartpole" if "statevector" in r.arm else "su2_cartpole"
    return "classic_dqn"


def local_runs(runs: pd.DataFrame) -> pd.DataFrame:
    d = runs[runs.log_origin == "local"].copy()
    t = pd.to_datetime(d.started_utc, utc=True)
    d["start"] = (t - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    d["end"] = d.start + d.wall_seconds
    return d


def mean_concurrency(d: pd.DataFrame) -> pd.Series:
    ev = sorted([(s, 1) for s in d.start] + [(e, -1) for e in d.end])
    times, cur = np.array([t for t, _ in ev]), np.cumsum([x for _, x in ev])

    def mean_conc(s, e):
        i0 = np.searchsorted(times, s, side="right") - 1
        i1 = np.searchsorted(times, e, side="left")
        tt = np.concatenate([[s], times[max(i0 + 1, 0):i1], [e]])
        cc = np.concatenate([[cur[i0] if i0 >= 0 else 0], cur[max(i0 + 1, 0):i1]])[: len(tt) - 1]
        dt = np.diff(tt)
        return float((cc * dt).sum() / dt.sum()) if dt.sum() > 0 else 1.0

    return pd.Series([mean_conc(s, e) for s, e in zip(d.start, d.end)], index=d.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/runs.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = local_runs(pd.read_csv(a.runs))
    d["mean_concurrency"] = mean_concurrency(d)
    d["hours"] = d.wall_seconds / 3600
    d["calibration_kind"] = [kind_of_run(r) for r in d.itertuples()]
    d[["config", "stage", "run", "calibration_kind", "hours", "start", "end", "mean_concurrency"]].to_csv(a.out, index=False)
    print(len(d), "local runs;", round(d.hours.sum(), 1), "run-hours")
    g = d.groupby("calibration_kind")
    print((g.apply(lambda x: (x.mean_concurrency * x.hours).sum() / x.hours.sum())).round(1).to_string())


if __name__ == "__main__":
    main()
