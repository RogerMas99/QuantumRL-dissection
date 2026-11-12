"""The energy estimate (docs/footprint/) is regenerated exactly from the committed calibration table and results/runs.csv."""
import pathlib
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "footprint"


def test_extrapolation_regenerates(tmp_path):
    subprocess.run([sys.executable, str(SCRIPTS / "summarize.py"), "--calibration-table", str(ROOT / "docs/footprint/calibration_table.csv"),
                    "--runs", str(ROOT / "results/runs.csv"), "--out", str(tmp_path)], check=True, capture_output=True)
    for name in ("extrapolation_by_type.csv", "extrapolation_totals.csv"):
        got, want = pd.read_csv(tmp_path / name), pd.read_csv(ROOT / "docs/footprint" / name)
        pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-9)


def test_hours_are_those_of_all_runs():
    runs = pd.read_csv(ROOT / "results/runs.csv")
    tot = pd.read_csv(ROOT / "docs/footprint/extrapolation_totals.csv").set_index("where").hours
    assert len(runs) == 6368
    assert abs(tot["ALL"] - runs.wall_seconds.sum() / 3600) < 2e-3
    assert abs(tot["this_PC"] + tot["colab"] - tot["ALL"]) < 2e-3
