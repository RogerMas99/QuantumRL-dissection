"""The control that replicates its input (superseded): matched_control_replicated{R}.

What must not silently drift: that the arm replicates R times (not 4), has a ReLU hidden layer, sees the full
observation, matches the hybrid's MEASURED total; that R = 4 is exactly matched_control_reuse4 (so it is not
re-run); that the name of the matched control cannot swallow the new one; and that the configuration runs the protocol of the reuse sweep.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from quantumrl_dissection.agents.capacity import build_agent_for, count_trainable  # noqa: E402
from quantumrl_dissection.arms import (MATCHED_INDICES_FULL, build_arm_config,  # noqa: E402
                                         hybrid_reuse_config)

CP = "CartPole-v1"
HYBRID_TOTAL = {4: 222, 8: 350, 16: 606, 32: 1118}          # measured, quoted in docs/adjustments.md


def _params(agent_type, cfg):
    return count_trainable(build_agent_for(agent_type, cfg, env_id=CP, is_qnet=True))


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_replicates_r_times_with_relu_hidden_layer_and_full_observation(R):
    at, cfg = build_arm_config(f"matched_control_replicated{R}", env_id=CP)
    assert at == "classic"
    assert cfg["n_repeats"] == R
    assert cfg["reuse_indices"] == MATCHED_INDICES_FULL == [0, 1, 2, 3]
    assert cfg["activation"] is nn.ReLU
    assert len(cfg["net_arch"]) == 1 and cfg["net_arch"][0] >= 1
    assert cfg["_matched_to"] == f"hybrid_reuse{R}"


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_budget_is_the_hybrids_measured_total_and_params_are_close(R):
    at, cfg = build_arm_config(f"matched_control_replicated{R}", env_id=CP)
    assert cfg["_budget"] == cfg["_hybrid_total"] == HYBRID_TOTAL[R]
    assert _params("hybrid", hybrid_reuse_config(R)) == HYBRID_TOTAL[R]
    measured = _params(at, cfg)
    assert measured == cfg["_spent_params"]
    assert abs(measured - HYBRID_TOTAL[R]) <= 0.07 * HYBRID_TOTAL[R]      # closest width, +/- a few units


def test_widths_are_the_ones_the_design_document_quotes():
    widths = {R: build_arm_config(f"matched_control_replicated{R}", env_id=CP)[1]["net_arch"][0] for R in HYBRID_TOTAL}
    assert widths == {4: 12, 8: 10, 16: 9, 32: 9}


@pytest.mark.parametrize("R", [8, 16, 32])
def test_replication_leaves_less_budget_for_the_hidden_layer_than_the_fixed_4_control(R):
    old = build_arm_config(f"matched_control_reuse{R}", env_id=CP)[1]
    new = build_arm_config(f"matched_control_replicated{R}", env_id=CP)[1]
    assert old["n_repeats"] == 4 and new["n_repeats"] == R
    assert new["net_arch"][0] < old["net_arch"][0]


def test_r4_is_exactly_the_existing_matched_r4_arm():
    keys = ("reuse_indices", "n_repeats", "net_arch", "activation", "_budget")
    old = build_arm_config("matched_control_reuse4", env_id=CP)[1]
    new = build_arm_config("matched_control_replicated4", env_id=CP)[1]
    assert {k: old[k] for k in keys} == {k: new[k] for k in keys}


def test_the_matched_control_names_do_not_swallow_each_other_and_garbage_is_rejected():
    for R in (8, 16, 32):
        cfg = build_arm_config(f"matched_control_replicated{R}", env_id=CP)[1]
        assert cfg["n_repeats"] == R                                     # not 4: it did not go to the matched control branch
    with pytest.raises(KeyError):
        build_arm_config("matched_control_replicatedXX", env_id=CP)
    with pytest.raises(KeyError):
        build_arm_config("matched_control_replicated", env_id=CP)


def test_configuration_runs_only_r8_16_32_with_the_protocol_of_the_reuse_sweep():
    import pathlib
    from quantumrl_dissection.training.experiment import config_cells, load_config
    cfg = load_config(pathlib.Path(__file__).resolve().parents[1] / "configs" / "rq3_output_reuse.toml")
    stage, cells = next((st, c) for st, c in config_cells(cfg, stage="replicated_control"))
    specs = [sp for _, sp in cells]
    assert len(specs) == 3 * 10
    assert {sp.arm for sp in specs} == {"matched_control_replicated8", "matched_control_replicated16", "matched_control_replicated32"}
    assert {sp.seed for sp in specs} == set(range(1, 11))
    assert all(sp.terminal_adjustment and sp.total_timesteps == 100_000 for sp in specs)
    assert all(sp.dqn_kwargs == {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10} for sp in specs)
    assert stage["eval_episodes"] == 5
