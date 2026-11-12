"""[policy] Simulator results are marked, detected and can be excluded from final tables.

docs/reproducibility.md: anything from the SU(2) emulator (agent_type su2) or the
statevector simulator (agent_type sv) is exploratory and must not enter the final work
without the author's explicit approval. These tests keep the marking from rotting.
"""
from __future__ import annotations

import json
import warnings

import pytest

pd = pytest.importorskip("pandas")

from quantumrl_dissection.analysis import loaders as analysis  # noqa: E402
from quantumrl_dissection.arms import ARMS  # noqa: E402
from quantumrl_dissection.training.provenance import (  # noqa: E402
    SIMULATED_AGENT_TYPES,
    exclude_simulated,
    is_simulated_arm,
    is_simulated_manifest,
    provenance_block,
)


def test_which_agent_types_are_simulators():
    assert SIMULATED_AGENT_TYPES == {"su2", "sv"}
    assert provenance_block("hybrid") is None and provenance_block("classic") is None
    blk = provenance_block("sv")
    assert blk["simulated"] is True and "SIMULATED" in blk["status"]


def test_every_su2_or_sv_arm_is_detected_and_no_real_arm_is():
    for name, (agent_type, _cfg) in ARMS.items():
        assert is_simulated_arm(name) == (agent_type in SIMULATED_AGENT_TYPES), name
    assert is_simulated_arm("sim_statevector_frozenlake_binary_depth5") and is_simulated_arm("sim_emulator_hybrid_unentangled")
    assert not is_simulated_arm("hybrid") and not is_simulated_arm("frozenlake_binary_depth5")


def test_manifest_detection_works_with_and_without_the_stamp():
    assert is_simulated_manifest({"agent_type": "su2"})                       # legacy, unstamped
    assert is_simulated_manifest({"agent_type": "hybrid", "provenance": {"simulated": True}})
    assert not is_simulated_manifest({"agent_type": "hybrid"})


def _write_run(outdir, name, agent_type):
    ep = outdir / f"{name}.csv"
    ep.write_text("episode_n,ep_reward,global_step\n" + "\n".join(f"{i},{10 + i},{i * 10}" for i in range(1, 80)))
    m = {"spec": {"arm": name, "seed": 1, "terminal_adjustment": True, "total_timesteps": 800},
         "run_name": name, "agent_type": agent_type,
         "outcome": {"episodes_csv": str(ep), "total_timesteps": 800, "wall_seconds": 1.0,
                     "probe": {"frac_poison": 0.0}}}
    (outdir / f"{name}.manifest.json").write_text(json.dumps(m))


def test_tables_carry_the_flag_warn_and_can_be_filtered(tmp_path):
    _write_run(tmp_path, "real_arm", "hybrid")
    _write_run(tmp_path, "sim_arm", "sv")
    with pytest.warns(UserWarning, match="SIMULATOR RESULTS PRESENT"):
        df = analysis.to_dataframe(tmp_path)
    assert dict(zip(df["arm"], df["simulated"])) == {"real_arm": False, "sim_arm": True}
    assert list(exclude_simulated(df)["arm"]) == ["real_arm"]               # default: excluded
    assert len(exclude_simulated(df, approved=True)) == 2                   # only on purpose


def test_no_warning_when_nothing_is_simulated(tmp_path):
    _write_run(tmp_path, "real_arm", "hybrid")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        analysis.to_dataframe(tmp_path)
