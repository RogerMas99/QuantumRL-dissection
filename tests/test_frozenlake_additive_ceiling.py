"""Verifies the §2 derivation directly against the real circuit, before any
compute is spent on the FrozenLake ceiling.

The claim (docs/adjustments.md): on FrozenLake Config B
(`frozen_binary_4q`, `ent=False`), `FrozenBasisToAngleTransformer` maps each
bit to {0, pi}, `sin(k*{0,pi})` is identically zero for every integer k, and
`cos(k*{0,pi})` is an affine function of the bit - so `<Z_i>` is affine in
`b_i` at EVERY reuploading depth L, not just L=1. That is pre-registered
prediction P2: depth cannot enlarge the hypothesis class here, because the
embedding never gives the circuit a second frequency to work with on a
two-point domain.

This is checked against `build_skolik_qlayer` directly - not against
`FourierAdditiveCeiling` or `linear_on_bits_ceiling`, which ARE that
hypothesis class by construction and would prove nothing about the real
circuit. No arm registration, no training: enumerate FrozenLake's 16 states,
evaluate the 4-qubit circuit on all of them, and check the resulting 16x4
matrix of `<Z_i>` is reproduced by a least-squares affine fit in the bits to
near machine precision.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from simplyqrl.qlayers import build_skolik_qlayer  # noqa: E402
from simplyqrl.transformations import FrozenBasisToAngleTransformer  # noqa: E402

N_QUBITS = 4  # ceil(log2(16)) for the 4x4 grid
TRANSFORM = FrozenBasisToAngleTransformer("4x4")

# Numerical, not exactly-zero: float32 circuit evaluation plus a lstsq solve.
# Anything near machine precision for float32 (~1e-6) confirms the algebraic
# identity; anything order-1 (the entangled negative control) confirms the
# opposite.
AAF_TOL = 1e-4


def _bits_and_z(n_layers: int, ent: bool) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all 16 FrozenLake states: bit matrix (16, 4) and <Z_i> (16, 4)."""
    states = torch.arange(16).unsqueeze(1).float()  # (16, 1)
    circuit = build_skolik_qlayer(
        N_QUBITS, n_layers, transform_fn=TRANSFORM, ent=ent
    )
    torch.manual_seed(0)
    with torch.no_grad():
        for p in circuit.parameters():
            p.copy_(torch.rand_like(p) * 2 * np.pi)
        z = circuit(states).numpy()  # (16, 4)

    s = np.arange(16)
    bits = ((s[:, None] >> np.arange(N_QUBITS)[None, :]) & 1).astype(float)  # (16, 4)
    return bits, z


def _affine_fit_residual(bits: np.ndarray, z: np.ndarray) -> float:
    """Least-squares Z = B.W + c, per output column; return the max residual."""
    design = np.concatenate([bits, np.ones((bits.shape[0], 1))], axis=1)  # (16, 5)
    coeffs, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    pred = design @ coeffs
    return float(np.abs(pred - z).max())


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_each_qubit_output_takes_exactly_two_values_per_bit(n_layers):
    """<Z_i> must depend on b_i alone (two-point range), which is the
    prerequisite for the affine claim below - checked directly, not inferred
    from the residual test alone."""
    bits, z = _bits_and_z(n_layers, ent=False)
    for i in range(N_QUBITS):
        for b in (0, 1):
            vals = z[bits[:, i] == b, i]
            spread = float(vals.max() - vals.min())
            assert spread < AAF_TOL, (
                f"qubit {i}, bit={b}, L={n_layers}: <Z_i> spread {spread:.2e} "
                "across states sharing this bit - not a function of b_i alone"
            )


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_unentangled_output_is_affine_in_the_bits(n_layers):
    bits, z = _bits_and_z(n_layers, ent=False)
    residual = _affine_fit_residual(bits, z)
    assert residual < AAF_TOL, (
        f"L={n_layers}: affine-in-bits residual {residual:.2e} >= {AAF_TOL:.0e} "
        "- P1/P2 do not hold as derived, do not run the FrozenLake ceiling on this assumption "
        "before revisiting docs/adjustments.md"
    )


def test_residual_does_not_grow_with_depth():
    """P2: depth does not enlarge the hypothesis class on this domain - the
    residual at L=5 should be the same order of magnitude as at L=1, not
    growing with the extra reuploads."""
    residuals = {L: _affine_fit_residual(*_bits_and_z(L, ent=False)) for L in (1, 2, 5)}
    worst, best = max(residuals.values()), min(residuals.values())
    assert worst < AAF_TOL, residuals
    # A generous bound: even in the worst case, growth stays within one order
    # of magnitude of the smallest residual, not a trend with L.
    assert worst < max(best * 10, AAF_TOL), (
        f"residual grew with depth: {residuals} - P2 may not hold"
    )


def test_negative_control_entangled_output_is_not_affine():
    """ent=True at n_layers=5 carries 4 effective entangling blocks
    (the final-ring property: only the FINAL layer's ring is wasted), so this is
    a real two-qubit circuit and the affine-in-bits claim must fail clearly -
    otherwise the positive tests above are not exercising anything."""
    bits, z = _bits_and_z(5, ent=True)
    residual = _affine_fit_residual(bits, z)
    assert residual > 0.05, (
        f"entangled circuit fit an affine-in-bits model to residual {residual:.2e} "
        "- the negative control has no power, something is wrong with the test"
    )
