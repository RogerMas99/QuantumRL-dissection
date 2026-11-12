"""Cross table of the runs by configuration and by row of table F.1, with the hours of every row.

    python scripts/reconcile_runs.py                 # markdown on the standard output
    python scripts/reconcile_runs.py --check         # also fail if a sum does not close

Every run of ``results/runs.csv`` is counted, whether or not its curves are in ``results/`` (``in_results``): table F.1 is
the cost of everything that was executed. The script checks that every run belongs to exactly one row of the table and that the
rows add up to the total.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "third_party"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quantumrl_dissection.analysis import tables_extra  # noqa: E402
from quantumrl_dissection.analysis.data import Results  # noqa: E402

THESIS_RUNS = [128, 131, 61, 230, 240, 130, 888, 2300, 130, 2130]    # table F.1 of the thesis, row by row
KINDS = [("real_circuit", "real circuit"), ("simulator", "simulators"), ("classical", "classical")]


def classify(runs: pd.DataFrame) -> pd.Series:
    """Index of the row of table F.1 of each run (-1 if none, -2 if more than one)."""
    stage, variant = runs["stage"].fillna(""), runs["variant"].fillna("")
    member = np.array([[bool(pred(c, s, v)) for c, s, v in zip(runs["config"], stage, variant)] for _, pred in tables_extra.F1_ROWS]).T
    n = member.sum(axis=1)
    return pd.Series(np.where(n == 1, member.argmax(axis=1), np.where(n == 0, -1, -2)), index=runs.index)


def build(results: pathlib.Path):
    runs = Results(results).runs().copy()
    runs["row"] = classify(runs)
    runs["hours"] = runs["wall_seconds"].astype(float) / 3600.0
    labels = [lab for lab, _ in tables_extra.F1_ROWS]
    return runs, labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    runs, labels = build(pathlib.Path(a.results))
    problems = []
    if (runs["row"] < 0).any():
        problems.append(f"{int((runs['row'] < 0).sum())} runs belong to no row or to several rows of table F.1")

    print("## Runs by configuration and row of table F.1\n")
    head = ["config"] + [f"F.1-{i + 1}" for i in range(len(labels))] + ["runs", "of which curves in `results/`"]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] * len(head)) + "|")
    for cfg, g in runs.groupby("config"):
        cells = [str(int((g["row"] == i).sum())) or "0" for i in range(len(labels))]
        cells = [c if c != "0" else "-" for c in cells]
        print(f"| {cfg} | " + " | ".join(cells) + f" | {len(g):,} | {int(g['in_results'].sum()):,} |")
    tot = [int((runs["row"] == i).sum()) for i in range(len(labels))]
    print("| **total** | " + " | ".join(f"**{t:,}**" for t in tot) + f" | **{len(runs):,}** | **{int(runs['in_results'].sum()):,}** |")
    print("| thesis, table F.1 | " + " | ".join(f"{t:,}" for t in THESIS_RUNS) + f" | {sum(THESIS_RUNS):,} | |")
    print()
    for i, (lab, t) in enumerate(zip(labels, THESIS_RUNS)):
        print(f"* F.1-{i + 1}: {lab}")
        if tot[i] != t:
            problems.append(f"row {i + 1} has {tot[i]} runs here and {t} in the thesis")
    if sum(tot) != len(runs):
        problems.append("the rows do not add up to the runs of the table")

    print("\n## Hours of table F.1, unrounded (wall-clock seconds of every run, divided by 3,600)\n")
    print("| row | real circuit | simulators | classical | total | runs | of which not in `results/` |")
    print("|---|---|---|---|---|---|---|")
    grand = np.zeros(3)
    for i, lab in enumerate(labels):
        g = runs[runs["row"] == i]
        h = [g.loc[g["kind"] == k, "hours"].sum() for k, _ in KINDS]
        grand += h
        print(f"| F.1-{i + 1} | " + " | ".join(f"{x:,.3f}" if x else "-" for x in h) + f" | {sum(h):,.3f} | {len(g):,} | {int((~g['in_results']).sum()):,} |")
    print(f"| **total** | " + " | ".join(f"**{x:,.3f}**" for x in grand) + f" | **{grand.sum():,.3f}** | **{len(runs):,}** | **{int((~runs['in_results']).sum()):,}** |")
    print()
    if not np.isclose(grand.sum(), runs["hours"].sum()):
        problems.append("the hours by kind do not add up to the hours of all the runs")
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    return 1 if (problems and a.check) else 0


if __name__ == "__main__":
    sys.exit(main())
