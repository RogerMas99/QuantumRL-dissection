"""The arms that run on the simulators of the circuit: registration, twins, parameter parity with the real arms, configurations.

Structural checks only; the numbers come from the runs. What must not silently drift: that every simulator arm builds with the expected
agent type and entanglement, that the unentangled arms of the two simulators are exact twins (same configuration, same parameter
count), that each replica has the parameter count of the real arm it stands in for, that the emulator family is unentangled by
construction, and that the configurations list the cells that the study quotes.
"""
from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")

from quantumrl_dissection.agents.capacity import build_agent_for, count_trainable  # noqa: E402
from quantumrl_dissection.arms import ARMS, build_arm_config, sim_hybrid_name  # noqa: E402
from quantumrl_dissection.environments import FROZEN_SCALAR_ID  # noqa: E402
from quantumrl_dissection.training.experiment import config_cells, load_config  # noqa: E402

CP = "CartPole-v1"
CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


def _params(arm, env=CP):
    at, cfg = build_arm_config(arm, env_id=env)
    return count_trainable(build_agent_for(at, cfg, env_id=env, is_qnet=True))


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_statevector_unentangled_arm_is_the_exact_twin_of_the_emulator_arm(R):
    sv, su = sim_hybrid_name("statevector", 5, R, unentangled=True), sim_hybrid_name("emulator", 5, R, unentangled=True)
    assert ARMS[sv][0] == "sv" and ARMS[su][0] == "su2"
    assert ARMS[sv][1] == ARMS[su][1]
    assert _params(sv) == _params(su)


@pytest.mark.parametrize("L", [1, 2])
def test_depth_unentangled_statevector_arm_is_the_twin_of_the_emulator_arm(L):
    sv, su = sim_hybrid_name("statevector", L, 1, unentangled=True), sim_hybrid_name("emulator", L, 1, unentangled=True)
    assert ARMS[sv][0] == "sv" and ARMS[su][0] == "su2"
    assert ARMS[sv][1] == ARMS[su][1] and ARMS[sv][1]["ent"] is False
    assert _params(sv) == _params(su)


def test_every_emulator_arm_is_unentangled_and_every_simulator_arm_is_flagged():
    from quantumrl_dissection.training.provenance import is_simulated_arm
    for name, (kind, cfg) in ARMS.items():
        if kind == "su2":
            assert not cfg.get("ent", False), name
        assert is_simulated_arm(name) == (kind in ("sv", "su2")), name
        assert name.startswith("sim_") == (kind in ("sv", "su2")), name


@pytest.mark.parametrize("name,kind,ent", [("sim_statevector_frozenlake_binary_depth5", "sv", True),
                                           ("sim_statevector_hybrid_reuse16", "sv", True),
                                           ("sim_emulator_hybrid_unentangled_reuse8", "su2", False)])
def test_arms_build_with_the_expected_agent_type_and_entanglement(name, kind, ent):
    env = CP if "frozenlake" not in name else FROZEN_SCALAR_ID
    agent_type, cfg = build_arm_config(name, env_id=env)
    assert agent_type == kind and bool(cfg["ent"]) is ent


def test_replicas_have_the_parameter_counts_of_the_real_arms():
    ref = {"sim_statevector_hybrid_reuse16": 606, "sim_emulator_hybrid_unentangled_reuse8": 350, "sim_statevector_hybrid": 126}
    for name, n in ref.items():
        assert _params(name) == n, name
    pairs = {"sim_statevector_porting_skolik_4q_depth1": "porting_skolik_4q_depth1", "sim_statevector_porting_skolik_4q_depth5": "porting_skolik_4q_depth5",
             "sim_statevector_porting_skolik_8q_depth2": "porting_skolik_8q_depth2",
             "sim_emulator_porting_skolik_4q_depth2_unentangled": "porting_skolik_4q_depth2_unentangled",
             "sim_statevector_hybrid_depth2_reuse8": "hybrid_depth2_reuse8", "sim_statevector_hybrid_depth1_reuse32": "hybrid_depth1_reuse32",
             "sim_statevector_hybrid_unentangled_reuse4": "hybrid_unentangled_reuse4"}
    for sim, real in pairs.items():
        assert _params(sim) == _params(real), (sim, real)


@pytest.mark.parametrize("name", ["sim_statevector_frozenlake_binary_depth1", "sim_statevector_frozenlake_binary_depth5",
                                  "sim_emulator_frozenlake_binary_unentangled_depth5", "sim_emulator_frozenlake_scalar_depth10"])
def test_frozenlake_replicas_have_the_parameter_counts_of_the_real_arms(name):
    real = name.replace("sim_statevector_", "").replace("sim_emulator_", "")
    assert _params(name, FROZEN_SCALAR_ID) == _params(real, FROZEN_SCALAR_ID)


def test_simulator_configurations_list_the_cells_of_the_study():
    val = load_config(CONFIGS / "simulator_validation.toml")
    assert sum(len(c) for _, c in config_cells(val)) == 410
    x100 = load_config(CONFIGS / "additional_x100.toml")
    assert sum(len(c) for _, c in config_cells(x100)) == 2300
    for cfg in (val, x100):
        assert cfg.get("simulated") is True
        for st, cells in config_cells(cfg):
            for _, spec in cells:
                assert ARMS[spec.arm][0] in ("sv", "su2"), spec.arm
                assert spec.terminal_adjustment and spec.total_timesteps == 100_000
        for st in cfg["stage"]:
            assert st["eval_episodes"] == 20


def test_additional_results_reuse_200_runs_that_exist_in_the_runs_table():
    import pandas as pd
    runs = pd.read_csv(CONFIGS.parent / "results" / "runs.csv", keep_default_na=False, na_values=[""])
    reused = runs[runs.also_used_in == "additional_x100"]
    assert len(reused) == 200
    assert set(reused.config) <= {"simulator_validation"} and reused.kind.eq("simulator").all()
