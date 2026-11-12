"""The classical twin of the hybrid: classical_twin[_reuse{R}] and classical_twin_unbounded[_reuse{R}].

What must not silently drift: that the head and the Output Reuse of the twin ARE the hybrid's (same classes, same
shapes, same initial weights under one seed), that the bounded extractor stays in [-1, 1] and the unbounded one does
not, that the parameter counts are the hybrid's + 6, that no PennyLane layer is left inside the twin, that the arm
names cannot capture each other or an existing arm, and that the configuration runs the protocol of the reuse sweep.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from torch import nn  # noqa: E402

import quantumrl_dissection  # noqa: E402,F401  (applies compat.py patches on import)
from quantumrl_dissection.agents.capacity import build_agent_for, count_trainable  # noqa: E402
from quantumrl_dissection.agents.classical_twin import ClassicalTwinAgent  # noqa: E402
from quantumrl_dissection.arms import ARMS, build_arm_config, hybrid_reuse_config, twin_name  # noqa: E402
from quantumrl_dissection.training.provenance import is_simulated_arm  # noqa: E402
from simplyqrl.agents import HybridAgent, OutputReuse  # noqa: E402

CP = "CartPole-v1"
RS = [1, 4, 8, 16, 32]
HYBRID_TOTAL = {1: 126, 4: 222, 8: 350, 16: 606, 32: 1118}        # measured
TWIN_TOTAL = {1: 132, 4: 228, 8: 356, 16: 612, 32: 1124}          # hybrid + 6
FAMS = ("classical_twin", "classical_twin_unbounded")


def _twin(fam, R, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    at, cfg = build_arm_config(twin_name(R, "unbounded" in fam), env_id=CP)
    return build_agent_for(at, cfg, env_id=CP, is_qnet=True)


def _hybrid(R, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return build_agent_for("hybrid", hybrid_reuse_config(R), env_id=CP, is_qnet=True)


def _signature(seq):
    return [(type(m).__name__, tuple(tuple(p.shape) for p in m.parameters()),
             getattr(m, "n_repeats", None)) for m in seq]


@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("R", RS)
def test_head_and_output_reuse_are_structurally_the_hybrids(fam, R):
    tw, hy = _twin(fam, R), _hybrid(R)
    assert isinstance(tw, ClassicalTwinAgent) and isinstance(tw, HybridAgent)
    assert _signature(tw.head) == _signature(hy.network[1:])
    names = [type(m).__name__ for m in tw.head]
    assert names == (["OutputReuse"] if R > 1 else []) + ["Linear", "Identity", "Linear"]
    if R > 1:
        assert isinstance(tw.head[0], OutputReuse) and tw.head[0].n_repeats == R
        assert tw.head[1].in_features == 8 * R
    else:
        assert tw.head[0].in_features == 8
    assert (tw.head[-3].out_features, tw.head[-1].out_features) == (4, 2)


@pytest.mark.parametrize("R", [1, 4, 32])
def test_head_starts_with_the_hybrids_weights_under_one_seed(R):
    tw, hy = _twin("classical_twin", R, seed=7), _hybrid(R, seed=7)
    for a, b in zip(tw.head.parameters(), hy.network[1:].parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("fam", FAMS)
def test_extractor_shape_and_layers(fam):
    tw = _twin(fam, 4)
    kinds = [type(m).__name__ for m in tw.extractor]
    assert kinds == ["Linear", "ReLU", "Linear"] + (["Tanh"] if fam == "classical_twin" else [])
    assert (tw.extractor[0].in_features, tw.extractor[0].out_features) == (4, 6)
    assert (tw.extractor[2].in_features, tw.extractor[2].out_features) == (6, 8)


def test_bounded_extractor_is_in_minus_one_one_and_unbounded_is_not():
    torch.manual_seed(0)
    x = torch.randn(512, 4) * 50.0                                    # far outside CartPole's range on purpose
    bounded, unbounded = _twin("classical_twin", 4, seed=0), _twin("classical_twin_unbounded", 4, seed=0)
    with torch.no_grad():
        zb, zu = bounded.extractor(x), unbounded.extractor(x)
    assert zb.shape == zu.shape == (512, 8)
    assert float(zb.min()) >= -1.0 and float(zb.max()) <= 1.0
    assert float(zu.abs().max()) > 1.0
    with torch.no_grad():                                             # also on CartPole-like observations
        assert float(bounded.extractor(torch.randn(64, 4)).abs().max()) <= 1.0


@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("R", RS)
def test_parameter_counts_are_the_hybrids_plus_six(fam, R):
    tw, hy = _twin(fam, R), _hybrid(R)
    assert count_trainable(hy) == HYBRID_TOTAL[R]
    assert count_trainable(tw) == TWIN_TOTAL[R] == count_trainable(hy) + 6
    assert count_trainable(tw.extractor) == 86 == (4 * 6 + 6) + (6 * 8 + 8)
    assert count_trainable(tw.head) == count_trainable(hy.network[1:]) == 32 * R + 4 + 10


def test_no_quantum_layer_is_left_and_no_parameter_is_registered_twice():
    tw = _twin("classical_twin", 8)
    assert not hasattr(tw, "qnet_q_layer")
    assert not any("q_layer" in n or "weights" == n.split(".")[-1] for n, _ in tw.named_parameters())
    keys = list(tw.state_dict())
    assert len(keys) == len(set(keys)) == 8                            # 4 Linear layers x (weight, bias)
    assert sum(p.numel() for p in tw.parameters()) == count_trainable(tw)


def test_forward_shape_and_gradients_reach_extractor_and_head():
    tw = _twin("classical_twin", 4, seed=1)
    out = tw(torch.randn(5, 4))
    assert out.shape == (5, 2)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in tw.parameters())
    assert any(float(p.grad.abs().sum()) > 0 for p in tw.extractor.parameters())


def test_only_dqn_and_only_hybrid_reuse_configs_are_accepted():
    at, cfg = build_arm_config("classical_twin_reuse4", env_id=CP)
    with pytest.raises(ValueError, match="is_qnet=True"):
        build_agent_for(at, cfg, env_id=CP, is_qnet=False)
    with pytest.raises(ValueError, match="skolik"):
        ClassicalTwinAgent(4, 2, {"net_arch": [4], "activation": nn.Identity, "n_qubits": 8}, is_qnet=True)


def test_registry_names_flags_and_no_prefix_capture():
    for R in RS:
        at, cfg = build_arm_config(twin_name(R), env_id=CP)
        au, cu = build_arm_config(twin_name(R, True), env_id=CP)
        assert at == au == "classical_twin"
        assert cfg["twin_bounded"] is True and cu["twin_bounded"] is False
        assert cfg["twin_hidden"] == cu["twin_hidden"] == 6
        assert cfg["reuse_repetitions"] == cu["reuse_repetitions"] == R
        # the head is defined by the hybrid_reuse{R} config: nothing but the two twin keys differs
        assert {k: v for k, v in cfg.items() if not k.startswith("twin_")} == hybrid_reuse_config(R)
    with pytest.raises(KeyError):
        build_arm_config("classical_twin_r3", env_id=CP)
    with pytest.raises(KeyError):
        build_arm_config("classical_twin_unbounded_reuse", env_id=CP)
    # existing arms untouched and not captured by the new names
    assert build_arm_config("matched_control_reuse8", env_id=CP)[1]["n_repeats"] == 4
    assert build_arm_config("matched_control_replicated8", env_id=CP)[1]["n_repeats"] == 8
    assert ARMS["hybrid"][0] == "hybrid" and ARMS["hybrid_reuse8"][0] == "hybrid"


def test_twin_is_not_flagged_as_a_simulator():
    assert not any(is_simulated_arm(twin_name(R, "unbounded" in f)) for f in FAMS for R in RS)


def test_configuration_runs_the_protocol_of_the_reuse_sweep_with_100_cells():
    import pathlib
    from quantumrl_dissection.training.experiment import config_cells, load_config
    cfg = load_config(pathlib.Path(__file__).resolve().parents[1] / "configs" / "rq2_capacity.toml")
    stage, cells = next((st, c) for st, c in config_cells(cfg, stage="classical_twin"))
    specs = [spec for _, spec in cells]
    assert len(specs) == 2 * 5 * 10 == 100
    assert {s.arm for s in specs} == {twin_name(R, u) for u in (False, True) for R in RS}
    assert {s.seed for s in specs} == set(range(1, 11))
    assert all(s.terminal_adjustment and s.total_timesteps == 100_000 for s in specs)
    assert all(s.dqn_kwargs == {"batch_size": 128, "buffer_size": 10_000, "train_frequency": 10} for s in specs)
    assert stage["eval_episodes"] == 5
    assert len({s.run_name for s in specs}) == 100


def test_parameter_split_matches_the_documented_numbers():
    from quantumrl_dissection.analysis.tables import _params_split
    for R in RS:
        for u in (False, True):
            total, (extractor, head) = _params_split(twin_name(R, u))
            assert total == TWIN_TOTAL[R] and extractor == 86 and head == 32 * R + 4 + 10
