"""Training-trajectory parity of the statevector agent and the real SimplyQRL agent.

tests/test_statevector_equivalence.py compares one call at a time. This compares a whole
optimisation: the real ENTANGLED `HybridAgent` (PennyLane circuit + Output Reuse + head) and
the `sv` agent are given identical initial weights and identical data, trained with the same
Adam for several steps, and their loss curves and final outputs must stay together. Because
the data and the initial weights are identical, there is no sampling noise: any divergence
beyond float rounding would be a bug in the simulator.

What this does NOT check is DQN's own stochasticity (epsilon-greedy draws, replay sampling):
that is statistical and is checked by running the same arms on both (the validation groups V and P,
criteria in docs/design/EXPERIMENT-09.md section 6).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from quantumrl_dissection.agents.capacity import build_agent_for  # noqa: E402
from quantumrl_dissection.arms import build_arm_config  # noqa: E402
from quantumrl_dissection.environments import FROZEN_SCALAR_ID  # noqa: E402

def _config(depth, reuse=1):
    from quantumrl_dissection.arms import frozenlake_binary_config
    cfg = frozenlake_binary_config(depth, ent=True)
    if reuse > 1:
        cfg["reuse_repetitions"] = reuse
    return cfg


CASES = [
    ("depth 3, output reuse x4", _config(3, 4), 40),                  # entangled, 3 layers, Output Reuse x4
    ("depth 5 (the arm that solves the task)", _config(5), 25),
    ("depth 2", _config(2), 40),
]


def _copy_weights(real, sv):
    for m_real, m_sv in zip(real.network, sv.network):
        if hasattr(m_sv, "load_weights_from_torchlayer"):
            m_sv.load_weights_from_torchlayer(m_real)
        elif any(True for _ in m_sv.parameters()):
            m_sv.load_state_dict(m_real.state_dict())


def _train(net, x, y, steps):
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


@pytest.mark.parametrize("label,cfg,steps", CASES)
def test_loss_curve_of_the_sv_agent_follows_the_real_entangled_agent(label, cfg, steps):
    from quantumrl_dissection.arms import resolve_transform
    cfg = resolve_transform(dict(cfg))
    assert cfg["ent"] is True
    torch.manual_seed(0)
    real = build_agent_for("hybrid", cfg, env_id=FROZEN_SCALAR_ID, is_qnet=True)
    sv = build_agent_for("sv", cfg, env_id=FROZEN_SCALAR_ID, is_qnet=True)
    _copy_weights(real, sv)

    torch.manual_seed(1)
    x = torch.arange(16).float().unsqueeze(1)
    y = torch.rand(16, 4)
    with torch.no_grad():
        assert float((real(x) - sv(x)).abs().max()) < 1e-5      # same function at step 0

    l_real, l_sv = _train(real, x, y, steps), _train(sv, x, y, steps)
    gap = max(abs(a - b) for a, b in zip(l_real, l_sv))
    assert gap < 1e-5, f"loss curves diverged: max gap {gap:.2e}"
    assert l_real[-1] < l_real[0], "the fit should actually be learning something"
    with torch.no_grad():
        assert float((real(x) - sv(x)).abs().max()) < 1e-4       # same function after training


def test_the_two_simulators_agree_with_each_other_when_unentangled():
    """sv with ent=False and the SU(2) emulator are two routes to the same circuit."""
    _, cfg = build_arm_config("sim_statevector_frozenlake_binary_unentangled_depth5", env_id=FROZEN_SCALAR_ID)
    assert cfg["ent"] is False
    torch.manual_seed(0)
    sv = build_agent_for("sv", cfg, env_id=FROZEN_SCALAR_ID, is_qnet=True)
    su2 = build_agent_for("su2", cfg, env_id=FROZEN_SCALAR_ID, is_qnet=True)
    su2.network[0].weights.data.copy_(sv.network[0].weights.data)
    for m_sv, m_su2 in zip(sv.network, su2.network):
        if not hasattr(m_sv, "weights") and any(True for _ in m_sv.parameters()):
            m_su2.load_state_dict(m_sv.state_dict())
    torch.manual_seed(1)
    x = torch.arange(16).float().unsqueeze(1)
    y = torch.rand(16, 4)
    with torch.no_grad():
        assert float((sv(x) - su2(x)).abs().max()) < 1e-6
    l_sv, l_su2 = _train(sv, x, y, 60), _train(su2, x, y, 60)
    assert max(abs(a - b) for a, b in zip(l_sv, l_su2)) < 1e-5
