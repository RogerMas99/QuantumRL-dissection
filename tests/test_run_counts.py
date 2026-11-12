"""Table F.1 counts every executed run once, discarded runs included, and agrees with the runs of the thesis row by row."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reconcile_runs", ROOT / "scripts" / "reconcile_runs.py")
reconcile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconcile)


@pytest.fixture(scope="module")
def runs():
    r, _ = reconcile.build(ROOT / "results")
    return r


def test_every_run_belongs_to_exactly_one_row(runs):
    assert (runs["row"] >= 0).all()


def test_runs_by_row_are_those_of_the_thesis(runs):
    got = [int((runs["row"] == i).sum()) for i in range(len(reconcile.THESIS_RUNS))]
    assert got == reconcile.THESIS_RUNS and len(runs) == 6368


def test_discarded_runs_are_counted(runs):
    assert int((~runs["in_results"]).sum()) == 540
    cell = runs[(runs["row"] == 0) & (runs["kind"] == "simulator")]
    assert len(cell) == 20 and int((~cell["in_results"]).sum()) == 10          # the ten of the zero initialization are in the cell
    assert cell["hours"].sum() == pytest.approx(1969.5 / 3600.0)


def test_hours_add_up(runs):
    by_kind = runs.groupby("kind")["hours"].sum()
    assert by_kind.sum() == pytest.approx(runs["hours"].sum())
    assert np.round(by_kind["real_circuit"], 1) == 3102.7 and np.round(by_kind["simulator"], 1) == 968.3 and np.round(by_kind["classical"], 1) == 56.3
