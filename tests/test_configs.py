"""The configurations list exactly the runs of the study, and the table of runs is consistent with them."""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from quantumrl_dissection.training.experiment import config_cells, load_config

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.toml"))
EXPECTED = {"rq1_porting": 240, "rq2_capacity": 140, "rq3_output_reuse": 160, "rq3_depth": 110, "rq3_entanglement": 80, "rq4_ceilings": 60,
            "rq5_frozenlake": 110, "lr_experiment": 2130, "terminal_ablation": 70, "simulator_validation": 410, "additional_x100": 2300,
            "pilot_60k": 18}


def test_there_is_one_configuration_per_research_question_and_experiment():
    assert {p.stem for p in CONFIGS} == set(EXPECTED)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_cells_of_each_configuration_are_the_runs_of_the_table(path):
    cfg = load_config(path)
    cells = [(st["name"], spec.run_name) for st, c in config_cells(cfg) for _, spec in c]
    assert len(cells) == EXPECTED[path.stem]
    assert len(set(cells)) == len(cells)
    runs = pd.read_csv(ROOT / "results" / "runs.csv", keep_default_na=False, na_values=[""])
    runs = runs[(runs.config == path.stem) & (runs.in_results == True)]               # noqa: E712
    table = {(r.stage, r.run) for r in runs.itertuples()}
    assert {(s, n) for s, n in cells} == table


def test_the_table_has_the_6368_runs_of_the_campaign():
    runs = pd.read_csv(ROOT / "results" / "runs.csv", keep_default_na=False, na_values=[""])
    assert len(runs) == 6368
    assert set(runs.role) == {"campaign", "probe"} and int((runs.role == "probe").sum()) == 2
    assert int(runs.in_results.sum()) == sum(EXPECTED.values()) - 0 == 5828
    assert not runs.duplicated(["config", "stage", "arm", "variant", "seed"]).any()


def test_every_configuration_of_a_simulator_is_flagged():
    for p in CONFIGS:
        assert bool(load_config(p).get("simulated", False)) == (p.stem in ("simulator_validation", "additional_x100"))
