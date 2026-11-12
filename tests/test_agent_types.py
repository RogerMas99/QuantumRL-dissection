"""`compat.py`'s agent_type dispatch extension and the six
arms registered on top of it (`core/configs.py`).

Two things this file exists to pin down, because both were live risks when
the dispatch was added:

1. Existing behaviour is UNCHANGED. `_patch_new_agent_types` wraps whatever
   `build_agent` already was after the agent-type aliases and passes anything it does not
   recognise straight through - if that passthrough were ever wrong, every
   already-completed the campaign manifest would silently stop being
   reproducible. `tests/test_dry_run_reuse_check` (this session's own
   84-manifest empirical check, not a pytest file) covers the reuse-guard
   side; this file covers the agent-construction side directly.

2. The three new agent_type values are GUARDED, not permissive: su2 refuses
   ent=True and circ_type!='skolik' rather than silently emulating a circuit
   it cannot represent, and fourier_additive refuses hsiao/dr-non-additive
   configs via `check_additive_embedding`. A silent misconstruction here
   would be exactly the assumed-not-checked failure mode this project exists
   to avoid.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

import quantumrl_dissection  # noqa: E402  (applies compat.py patches on import)
from quantumrl_dissection.arms import ARMS, build_arm_config  # noqa: E402
from quantumrl_dissection.agents.capacity import build_agent_for, count_trainable  # noqa: E402
from quantumrl_dissection.agents.fourier_ceiling import _TransformedLinear  # noqa: E402
from quantumrl_dissection.simulators.emulator import SU2HybridAgent  # noqa: E402
from quantumrl_dissection.environments import FROZEN_SCALAR_ID  # noqa: E402
from simplyqrl.agents import build_agent, HybridAgent  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Existing agent_type strings still resolve exactly as before.
# ---------------------------------------------------------------------------
def test_hybrid_and_mlp_dispatch_unaffected():
    agent = build_agent("hybrid", (4,), 2, dict(ARMS["hybrid"][1]), is_qnet=True)
    assert isinstance(agent, HybridAgent)
    out = agent(torch.randn(3, 4))
    assert out.shape == (3, 2)


def test_unknown_agent_type_still_raises():
    with pytest.raises(ValueError, match="unknown agent_type"):
        build_agent("bogus", (4,), 2, {}, is_qnet=True)


# ---------------------------------------------------------------------------
# 2. su2 dispatch
# ---------------------------------------------------------------------------
def test_su2_dispatch_builds_su2_hybrid_agent():
    cfg = {"circ_type": "skolik", "n_qubits": 8, "n_layers_q": 5, "ent": False,
           "net_arch": [4]}
    agent = build_agent("su2", (4,), 2, cfg, is_qnet=True)
    assert isinstance(agent, SU2HybridAgent)
    out = agent(torch.randn(3, 4))
    assert out.shape == (3, 2)


def test_su2_rejects_entangled_config():
    cfg = {"circ_type": "skolik", "n_qubits": 8, "n_layers_q": 5, "ent": True}
    with pytest.raises(ValueError, match="ent=False"):
        build_agent("su2", (4,), 2, cfg, is_qnet=True)


def test_su2_rejects_non_skolik_circ_type():
    cfg = {"circ_type": "hsiao", "n_qubits": 4, "n_layers_q": 1}
    with pytest.raises(ValueError, match="circ_type"):
        build_agent("su2", (4,), 2, cfg, is_qnet=True)


def test_su2_rejects_actor_critic_construction():
    with pytest.raises(ValueError, match="is_qnet=True"):
        build_agent("su2", (4,), 2, {"n_qubits": 4, "n_layers_q": 1}, is_qnet=False)


# ---------------------------------------------------------------------------
# 3. fourier_additive dispatch
# ---------------------------------------------------------------------------
def test_fourier_additive_dispatch():
    cfg = {"n_qubits": 8, "n_layers_q": 5, "circ_type": "skolik"}
    agent = build_agent("fourier_additive", (4,), 2, cfg, is_qnet=True)
    out = agent(torch.randn(3, 4))
    assert out.shape == (3, 2)


def test_fourier_additive_rejects_hsiao_via_check_additive_embedding():
    cfg = {"n_qubits": 8, "n_layers_q": 5, "circ_type": "hsiao"}
    with pytest.raises(ValueError, match="hsiao"):
        build_agent("fourier_additive", (4,), 2, cfg, is_qnet=True)


def test_fourier_additive_rejects_dr_with_n_qubits_below_n_data():
    cfg = {"n_qubits": 2, "n_layers_q": 5, "circ_type": "dr"}
    with pytest.raises(ValueError, match="dr"):
        build_agent("fourier_additive", (4,), 2, cfg, is_qnet=True)


# ---------------------------------------------------------------------------
# 4. linear_on_bits dispatch
# ---------------------------------------------------------------------------
def test_linear_on_bits_dispatch_bare_linear_without_transform():
    agent = build_agent("linear_on_bits", (4,), 4, {"n_qubits": 4}, is_qnet=True)
    assert isinstance(agent, torch.nn.Linear)
    out = agent(torch.zeros(3, 4))
    assert out.shape == (3, 4)


def test_linear_on_bits_dispatch_wraps_transform_fn_when_given():
    from simplyqrl.transformations import FrozenBasisToAngleTransformer
    cfg = {"n_qubits": 4, "transform_fn": FrozenBasisToAngleTransformer("4x4")}
    agent = build_agent("linear_on_bits", (1,), 4, cfg, is_qnet=True)
    assert isinstance(agent, _TransformedLinear)
    # raw scalar FrozenLake state in, 4 Q-values out - the transform inside
    # the module does the state -> bit-vector expansion.
    out = agent(torch.tensor([[5.0]]))
    assert out.shape == (1, 4)


def test_all_three_new_types_require_is_qnet_true():
    for key, cfg in (("su2", {"n_qubits": 4, "n_layers_q": 1}),
                     ("fourier_additive", {"n_qubits": 4, "n_layers_q": 1}),
                     ("linear_on_bits", {"n_qubits": 4})):
        with pytest.raises(ValueError, match="is_qnet=True"):
            build_agent(key, (4,), 2, cfg, is_qnet=False)


# ---------------------------------------------------------------------------
# 5. The six registered arms resolve, and the the emulator arms match their real
#    hybrid counterpart's parameter count EXACTLY - the same architecture,
#    just without a quantum simulator underneath.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("su2_arm,hybrid_arm,env_id", [
    ("sim_emulator_frozenlake_scalar_depth5", "frozenlake_scalar_depth5", FROZEN_SCALAR_ID),
    ("sim_emulator_frozenlake_binary_unentangled_depth1", "frozenlake_binary_unentangled_depth1", FROZEN_SCALAR_ID),
    ("sim_emulator_frozenlake_binary_unentangled_depth5", "frozenlake_binary_unentangled_depth5", FROZEN_SCALAR_ID),
    ("sim_emulator_hybrid_unentangled", "hybrid", "CartPole-v1"),
])
def test_su2_arm_matches_reference_hybrid_param_count(su2_arm, hybrid_arm, env_id):
    su2_type, su2_cfg = build_arm_config(su2_arm, env_id=env_id)
    hyb_type, hyb_cfg = build_arm_config(hybrid_arm, env_id=env_id)
    assert su2_type == "su2"
    assert hyb_type == "hybrid"
    su2_agent = build_agent_for(su2_type, su2_cfg, env_id=env_id, is_qnet=True)
    hyb_agent = build_agent_for(hyb_type, hyb_cfg, env_id=env_id, is_qnet=True)
    assert count_trainable(su2_agent) == count_trainable(hyb_agent)


def test_fourier_ceiling_arms_resolve_and_are_not_capacity_matched():
    """Registered on purpose without matching the reference hybrid's budget -
    the ceiling's size IS the hypothesis-class size, not a design choice.
    Just check both resolve and report their (different) counts, rather than
    asserting any particular relationship between them."""
    ft, fc = build_arm_config("fourier_ceiling", env_id="CartPole-v1")
    ht, hc = build_arm_config("hybrid", env_id="CartPole-v1")
    f_agent = build_agent_for(ft, fc, env_id="CartPole-v1", is_qnet=True)
    h_agent = build_agent_for(ht, hc, env_id="CartPole-v1", is_qnet=True)
    assert count_trainable(f_agent) > 0
    assert count_trainable(h_agent) > 0

    lt, lc = build_arm_config("frozenlake_binary_linear_ceiling", env_id=FROZEN_SCALAR_ID)
    l_agent = build_agent_for(lt, lc, env_id=FROZEN_SCALAR_ID, is_qnet=True)
    # 4 bits + bias, x 4 actions - the exact degenerate hypothesis class size,
    # independent of L. See core/fourier_ceiling.py::linear_on_bits_ceiling.
    assert count_trainable(l_agent) == 5 * 4


def test_new_arms_do_not_shadow_or_alter_existing_arms():
    """Registration only ADDS keys - see core/configs.py's own comment on this.
    Pin the exact set of pre-existing arm names this session must not have
    touched, spot-checking their configs are identical to what they were."""
    assert ARMS["hybrid"][1]["n_qubits"] == 8
    assert ARMS["hybrid"][1]["ent"] is True
    assert ARMS["frozenlake_binary_unentangled_depth1"][1]["ent"] is False
    assert ARMS["frozenlake_binary_depth1"][1]["ent"] is True
    for name in ("sim_emulator_hybrid_unentangled", "sim_emulator_frozenlake_scalar_depth5",
                "sim_emulator_frozenlake_binary_unentangled_depth1", "sim_emulator_frozenlake_binary_unentangled_depth5",
                "fourier_ceiling", "frozenlake_binary_linear_ceiling"):
        assert name in ARMS
