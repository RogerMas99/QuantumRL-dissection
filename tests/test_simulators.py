"""The simulators reproduce the real circuit: the same output and the same gradients on the same weights."""
import pytest
import torch
from simplyqrl.qlayers import build_skolik_qlayer
from simplyqrl.transformations import (
    CartPoleNormalizationTransformer, FrozenBasisToAngleTransformer, FrozenNormalizationTransformer,
)

from quantumrl_dissection.simulators import SU2SkolikEmulator, SVSkolikLayer

TOL = 1e-6
STATES = lambda b: torch.randint(0, 16, (b, 1)).float()  # noqa: E731
CASES = [
    ("cartpole_8q_depth5", 8, 5, CartPoleNormalizationTransformer(), lambda b: torch.rand(b, 4) * 2 - 1),
    ("frozenlake_binary_depth1", 4, 1, FrozenBasisToAngleTransformer("4x4"), STATES),
    ("frozenlake_binary_depth5", 4, 5, FrozenBasisToAngleTransformer("4x4"), STATES),
    ("frozenlake_scalar_depth5", 1, 5, FrozenNormalizationTransformer("4x4"), STATES),
]


def _pair(simulator, n_qubits, n_layers, transform_fn, ent=False, **kw):
    torch.manual_seed(0)
    real = build_skolik_qlayer(n_qubits, n_layers, transform_fn=transform_fn, ent=ent, **kw)
    sim = SVSkolikLayer(n_qubits, n_layers, transform_fn=transform_fn, ent=ent, **kw) if simulator == "sv" else \
        SU2SkolikEmulator(n_qubits, n_layers, transform_fn=transform_fn, **kw)
    sim.load_weights_from_torchlayer(real)
    return real, sim


def _agree(real, sim, x):
    """Largest difference of the outputs and of the gradients with respect to the weights."""
    out_real, out_sim = real(x), sim(x)
    weights = torch.randn_like(out_real)
    (out_real * weights).sum().backward()
    (out_sim * weights).sum().backward()
    return (out_real - out_sim).abs().max().item(), (next(real.parameters()).grad - sim.weights.grad).abs().max().item()


@pytest.mark.parametrize("label,n_qubits,n_layers,transform_fn,sampler", CASES)
def test_emulator_matches_the_unentangled_circuit(label, n_qubits, n_layers, transform_fn, sampler):
    real, emu = _pair("su2", n_qubits, n_layers, transform_fn)
    out, grad = _agree(real, emu, sampler(16))
    assert out < TOL and grad < 1e-5, (label, out, grad)


@pytest.mark.parametrize("ent", [False, True])
@pytest.mark.parametrize("label,n_qubits,n_layers,transform_fn,sampler", CASES)
def test_statevector_matches_the_circuit(label, n_qubits, n_layers, transform_fn, sampler, ent):
    if ent and n_qubits == 1:
        pytest.skip("a ring of CZ gates needs two qubits")
    real, sv = _pair("sv", n_qubits, n_layers, transform_fn, ent=ent)
    out, grad = _agree(real, sv, sampler(16))
    assert out < TOL and grad < 1e-5, (label, ent, out, grad)


def test_selected_components_are_embedded_like_the_real_circuit():
    real, emu = _pair("su2", 3, 2, None, emb_indices=[1, 2, 3])
    x = torch.rand(16, 4) * 2 - 1
    assert (real(x) - emu(x)).abs().max().item() < TOL


def test_the_emulator_differs_from_an_entangled_circuit():
    """Negative control: the agreement above is not trivial, because entanglement changes the circuit at depth > 1."""
    torch.manual_seed(0)
    tf = FrozenBasisToAngleTransformer("4x4")
    entangled = build_skolik_qlayer(4, 5, transform_fn=tf, ent=True)
    emu = SU2SkolikEmulator(4, 5, transform_fn=tf)
    emu.load_weights_from_torchlayer(entangled)
    x = STATES(16)
    assert (entangled(x) - emu(x)).abs().max().item() > 1e-3


def test_entanglement_is_a_no_op_at_depth_one():
    tf = FrozenBasisToAngleTransformer("4x4")
    with_cz, without = SVSkolikLayer(4, 1, transform_fn=tf, ent=True), SVSkolikLayer(4, 1, transform_fn=tf, ent=False)
    without.load_state_dict(with_cz.state_dict())
    x = STATES(16)
    assert (with_cz(x) - without(x)).abs().max().item() < TOL


def test_one_qubit_entanglement_is_rejected():
    with pytest.raises(ValueError):
        SVSkolikLayer(1, 3, ent=True)
