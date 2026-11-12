"""`SVSkolikLayer` must reproduce `build_skolik_qlayer(ent=...)`, entangled or not.

Same discipline as tests/test_su2_equivalence.py, with the one thing that file cannot do:
the ENTANGLED circuit. Forward output and gradients are compared with the real PennyLane
`TorchLayer` on the same weights to ~1e-6, on the configurations the project actually runs
(CartPole 8q L5, FrozenLake Config B at several depths, Config A at one qubit), plus the
n = 2 corner where the circular CZ ring applies CZ(0, 1) and CZ(1, 0) and cancels.

Two negative controls keep the positive checks honest:
- the entangled statevector layer must DIFFER substantially from the unentangled real
  circuit (otherwise "ent" would be a no-op in both and the comparison proves nothing);
- at L = 1 it must agree with the unentangled circuit (the final-ring property: the final CZ ring cannot
  affect a PauliZ readout, so L = 1 entangled and unentangled are the same circuit).

Plus the initialisation check of the emulator initialisation: weights ~ U[0, 2*pi).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from simplyqrl.qlayers import build_skolik_qlayer  # noqa: E402
from simplyqrl.transformations import (  # noqa: E402
    CartPoleNormalizationTransformer,
    FrozenBasisToAngleTransformer,
    FrozenNormalizationTransformer,
)

from quantumrl_dissection.simulators.statevector import SVSkolikLayer  # noqa: E402

TOL = 1e-6

CASES = [
    ("skolik_8q_cartpole_L5", 8, 5, CartPoleNormalizationTransformer(),
     lambda b: torch.rand(b, 4) * 2 - 1),
    ("frozen_binary_4q_L2", 4, 2, FrozenBasisToAngleTransformer("4x4"),
     lambda b: torch.randint(0, 16, (b, 1)).float()),
    ("frozen_binary_4q_L3", 4, 3, FrozenBasisToAngleTransformer("4x4"),
     lambda b: torch.randint(0, 16, (b, 1)).float()),
    ("frozenlake_binary_depth5", 4, 5, FrozenBasisToAngleTransformer("4x4"),
     lambda b: torch.randint(0, 16, (b, 1)).float()),
    ("two_qubits_L3", 2, 3, FrozenBasisToAngleTransformer("4x4"),
     lambda b: torch.randint(0, 16, (b, 1)).float()),
]


def _pair(n_qubits, n_layers, transform_fn, ent, seed=0):
    torch.manual_seed(seed)
    real = build_skolik_qlayer(n_qubits, n_layers, transform_fn=transform_fn, ent=ent)
    sv = SVSkolikLayer(n_qubits, n_layers, transform_fn=transform_fn, ent=ent)
    sv.load_weights_from_torchlayer(real)
    return real, sv


@pytest.mark.parametrize("label,n_qubits,n_layers,transform_fn,sampler", CASES)
@pytest.mark.parametrize("ent", [True, False])
def test_forward_agrees(label, n_qubits, n_layers, transform_fn, sampler, ent):
    real, sv = _pair(n_qubits, n_layers, transform_fn, ent)
    x = sampler(16)
    with torch.no_grad():
        a, b = real(x), sv(x)
    assert a.shape == b.shape
    assert float((a - b).abs().max()) < TOL, (label, ent)


@pytest.mark.parametrize("label,n_qubits,n_layers,transform_fn,sampler", CASES)
@pytest.mark.parametrize("ent", [True, False])
def test_gradient_wrt_weights_agrees(label, n_qubits, n_layers, transform_fn, sampler, ent):
    real, sv = _pair(n_qubits, n_layers, transform_fn, ent)
    x = sampler(8)
    wgt = torch.randn(x.shape[0], n_qubits)
    (real(x) * wgt).sum().backward()
    (sv(x) * wgt).sum().backward()
    g_real, g_sv = real.weights.grad, sv.weights.grad
    assert float((g_real - g_sv).abs().max()) < TOL, (label, ent)


def test_entangled_layer_differs_from_unentangled_real_circuit():
    """Negative control: with ent=True at L=5 the circuit must really be different."""
    tf = FrozenBasisToAngleTransformer("4x4")
    real_noent, _ = _pair(4, 5, tf, ent=False)
    torch.manual_seed(0)
    sv_ent = SVSkolikLayer(4, 5, transform_fn=tf, ent=True)
    sv_ent.load_weights_from_torchlayer(real_noent)
    x = torch.randint(0, 16, (32, 1)).float()
    with torch.no_grad():
        gap = float((real_noent(x) - sv_ent(x)).abs().max())
    assert gap > 0.05, gap


def test_depth_one_entangled_equals_unentangled():
    """the final-ring property: at L = 1 the final CZ ring cannot change a PauliZ readout."""
    tf = FrozenBasisToAngleTransformer("4x4")
    torch.manual_seed(0)
    a = SVSkolikLayer(4, 1, transform_fn=tf, ent=True)
    b = SVSkolikLayer(4, 1, transform_fn=tf, ent=False)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 16, (32, 1)).float()
    with torch.no_grad():
        assert float((a(x) - b(x)).abs().max()) < 1e-9


def test_one_qubit_entanglement_is_rejected():
    with pytest.raises(ValueError):
        SVSkolikLayer(1, 3, ent=True)


def test_initial_weights_follow_the_real_layer_distribution():
    torch.manual_seed(0)
    real = build_skolik_qlayer(8, 50, ent=True).weights.detach()
    torch.manual_seed(1)
    sv = SVSkolikLayer(8, 50, ent=True).weights.detach()
    assert tuple(sv.shape) == tuple(real.shape)
    for w in (real, sv):
        assert float(w.min()) >= 0.0 and float(w.max()) <= 2 * np.pi + 1e-6
        assert abs(float(w.mean()) - np.pi) < 0.2
        assert abs(float(w.std()) - 2 * np.pi / np.sqrt(12)) < 0.15


# ---------------------------------------------------------------------------
# Exploratory options: they must be inert by default and do what they say.
# ---------------------------------------------------------------------------
def test_options_are_off_by_default_and_trainable_scale_starts_as_identity():
    tf = FrozenBasisToAngleTransformer("4x4")
    torch.manual_seed(0)
    plain = SVSkolikLayer(4, 3, transform_fn=tf, ent=True)
    scaled = SVSkolikLayer(4, 3, transform_fn=tf, ent=True, trainable_input_scale=True)
    scaled.weights.data.copy_(plain.weights.data)
    x = torch.randint(0, 16, (16, 1)).float()
    assert not hasattr(plain, "input_scale")
    with torch.no_grad():
        assert float((plain(x) - scaled(x)).abs().max()) < 1e-12   # scale = 1 -> same circuit
    scaled(x).sum().backward()
    assert scaled.input_scale.grad is not None and float(scaled.input_scale.grad.abs().sum()) > 0


def test_small_normal_init_option():
    torch.manual_seed(0)
    layer = SVSkolikLayer(4, 50, ent=True, init_std=0.1)
    assert float(layer.weights.abs().max()) < 0.6 and abs(float(layer.weights.std()) - 0.1) < 0.03
