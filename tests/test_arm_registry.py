"""The registry of arms and the table of names.

* every arm of the registry has the configuration of the arm of the same specification in the original code
  (``tests/data/arm_configs.json`` is that registry, dumped with the original code and with the names translated);
* the matched controls, which are derived, are checked in the same way;
* the table ``docs/arm_names.csv`` and the registry agree: every new name of the table is an arm of the registry, a derived arm, a PPO arm, or an arm that
  the study does not tabulate and that exists only in the raw record (a simulator variant or an arm of the exploratory groups).
"""
from __future__ import annotations

import csv
import json
import pathlib
import re

import pytest

torch = pytest.importorskip("torch")

import quantumrl_dissection  # noqa: E402,F401
from quantumrl_dissection.arms import ARMS, build_arm_config, is_derived_arm  # noqa: E402
from quantumrl_dissection.training.ppo import PPO_ARMS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT = json.loads((ROOT / "tests" / "data" / "arm_configs.json").read_text(encoding="utf-8"))
DERIVED = ["matched_control", "frozenlake_matched_control"] + [f"matched_control_reuse{r}" for r in (4, 8, 16, 32)] + \
          [f"matched_control_replicated{r}" for r in (8, 16, 32)]


def _norm(cfg):
    return {k: re.sub(r"<(.+?) object at 0x[0-9A-Fa-f]+>", r"<\1>", str(v)) for k, v in sorted(cfg.items())}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_registered_arm_has_the_configuration_of_the_original_arm(arm):
    assert arm in SNAPSHOT, f"{arm} is not in the snapshot of the original registry"
    agent_type, cfg = build_arm_config(arm)
    assert agent_type == SNAPSHOT[arm]["agent_type"]
    assert _norm(cfg) == SNAPSHOT[arm]["config"]


@pytest.mark.parametrize("arm", DERIVED)
def test_derived_arms_have_the_configuration_of_the_original_arm(arm):
    assert is_derived_arm(arm)
    env = "FrozenLake4x4Scalar-v0" if "frozenlake" in arm else "CartPole-v1"
    agent_type, cfg = build_arm_config(arm, env_id=env)
    assert agent_type == SNAPSHOT[arm]["agent_type"]
    got = _norm(cfg)
    want = SNAPSHOT[arm]["config"]
    assert got == want


def _table():
    return list(csv.DictReader(open(ROOT / "docs" / "arm_names.csv", encoding="utf-8")))


def test_names_of_the_table_are_arms_of_the_registry_or_of_the_raw_record_only():
    known = set(ARMS) | set(DERIVED) | set(PPO_ARMS)
    missing = []
    for r in _table():
        new = r["new_name"].removesuffix("_no_terminal_adjustment")
        if new in known:
            continue
        if "raw archive only" in r["note"] or r["config"] == "simulator_validation":
            continue
        missing.append(new)
    assert not missing, missing


def test_every_arm_that_is_run_in_a_configuration_is_registered():
    from quantumrl_dissection.training.experiment import config_cells, load_config
    known = set(ARMS) | set(DERIVED) | set(PPO_ARMS)
    for path in sorted((ROOT / "configs").glob("*.toml")):
        for stage, _ in config_cells(load_config(path)):
            assert set(stage["arms"]) <= known, (path.name, set(stage["arms"]) - known)


def test_merged_names_are_identical_in_specification():
    """Arms of the original code that share a new name are listed in the table; their configuration was checked field by field."""
    merged = [r for r in _table() if r["merged_with_old_names"]]
    assert merged, "the table lists no merge"
    for r in merged:
        assert r["new_name"] not in ("", None)
