"""Verifies the GENERAL Schuld-Sweke-Meyer claim behind the ceiling directly
against the real circuit, on a continuous domain - not just FrozenLake's
two-point {0, pi} degeneracy, which `test_frozenlake_additive_ceiling.py`
covers separately.

The claim (docs/adjustments.md): for `build_skolik_qlayer(ent=False)`
with FIXED weights, `<Z_i>` as a function of the embedded feature x_i is
EXACTLY a degree-L trigonometric polynomial - L reuploads of a
single-generator RX rotation make frequencies 1..L accessible, no more and
no fewer.

Sampling design - and a bug this file used to have
-----------------------------------------------------
An earlier version of this test swept ONE shared x across ALL wires at once
(the same value on every qubit). That accidentally weakened the `ent=True`
negative control to a marginal ~1e-3: with every wire driven by the SAME
scalar, the whole system has only one true degree of freedom, so even the
ENTANGLED circuit collapses back to *some* Fourier series in that one x - a
higher-degree one, but still one the fit could partially absorb. Confirmed
directly: fitting the same entangled data with degree 10 instead of 5 nearly
zeroed the residual (0.19 -> 0.0053), which is not "not a Fourier series", it
is "a Fourier series of unexpectedly low degree, from too few independent
inputs to actually probe entanglement".

The method: sweep ONE wire's feature (x) and draw the OTHER wires' features
INDEPENDENTLY AT RANDOM per sample. For `ent=False` this must not matter at
all - the swept wire's `<Z>` is a function of x alone, by the emulator's
product-state argument, so unentangled circuits are unaffected (checked
directly below). For `ent=True`, the swept wire's state now genuinely
depends on its entangled neighbours' random, uncontrolled features, and NO
univariate function of x can capture that - which is what makes the negative
control decisive (median-across-seeds floor moved from ~1e-3 to ~0.5, three
orders of magnitude firmer) rather than marginal.

Robustness against an unlucky weight draw
--------------------------------------------
A single random weight draw can, by chance, give the top frequency k=L a
near-zero coefficient (observed directly: 1/30 draws in a wider calibration
sample landed under 1e-3 on the truncated-basis check, purely from the
Fourier coefficient at k=L happening to be small for that specific draw -
not a property of the circuit, a property of that one draw). A test that
checks only one seed can therefore pass or fail by luck. This file corrects
`N_DRAWS = 8` seeds, aggregates the four swept-qubit measurements per draw by
their MEAN (a single unlucky qubit should not swing a whole draw's verdict,
while an unlucky mean would still be informative), and asserts on the
MINIMUM across the 8 draws - the worst draw, not a chosen one.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from simplyqrl.qlayers import build_skolik_qlayer  # noqa: E402

N_QUBITS = 4
N_SAMPLES = 300
N_DRAWS = 8            # seeds 0..7, set - reproducible, not re-rolled per run
SEED_OFFSET = 3000      # for the independent "other wires" sampling, kept apart from weight seeds

# Calibrated against the actual 8-draw distribution (see the module docstring
# and docs/adjustments.md), not chosen to make the test pass narrowly:
#   positive (ent=False, full basis k<=L):      per-draw mean, worst of 8 draws = 1.3e-7
#   negative truncated (ent=False, k<=L-1):     per-draw mean, worst of 8 draws = 2.1e-2
#   negative entangled (ent=True,  k<=L):       per-draw mean, worst of 8 draws = 0.85 (min-across-qubits 0.49)
POSITIVE_TOL = 1e-4        # ~1000x margin above the observed worst case
TRUNCATED_FLOOR = 5e-3     # ~4x margin below the observed worst case
ENTANGLED_FLOOR = 0.2      # ~2.5x margin below the observed worst mean, ~2.5x below the worst single-qubit case too

X = np.linspace(-np.pi, np.pi, N_SAMPLES, endpoint=False)


def _relative_residual(z: np.ndarray, x: np.ndarray, k_max: int) -> float:
    """RMS(fit residual) / std(signal) - scale-free, so a wire whose output
    happens to have small amplitude does not get an easier bar to clear than
    one with large amplitude."""
    if k_max < 0:
        pred = np.full_like(z, z.mean())
    else:
        cols = [np.ones_like(x)]
        for k in range(1, k_max + 1):
            cols.append(np.cos(k * x))
            cols.append(np.sin(k * x))
        design = np.stack(cols, axis=1)
        coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
        pred = design @ coeffs
    rms_residual = float(np.sqrt(np.mean((pred - z) ** 2)))
    std_z = float(np.std(z))
    return rms_residual / max(std_z, 1e-12)


def _sample_sweep(circuit, sweep_qubit: int, others_seed: int) -> np.ndarray:
    """<Z_sweep_qubit> with x swept over X on that wire, and INDEPENDENT
    uniform random features on every other wire, resampled per data point -
    the design that makes the entangled negative control meaningful (see
    module docstring)."""
    rng = np.random.default_rng(others_seed)
    others = rng.uniform(-np.pi, np.pi, size=(N_SAMPLES, N_QUBITS - 1)).astype(np.float32)
    cols, oi = [], 0
    for q in range(N_QUBITS):
        if q == sweep_qubit:
            cols.append(X.astype(np.float32))
        else:
            cols.append(others[:, oi])
            oi += 1
    inputs = torch.tensor(np.stack(cols, axis=1))
    with torch.no_grad():
        z = circuit(inputs).numpy()
    return z[:, sweep_qubit]


def _per_draw_mean_residuals(n_layers: int, ent: bool, k_max: int) -> list[float]:
    """One value per weight draw: mean relative residual across all 4 wires
    swept in turn, with the other 3 independently randomised each time."""
    out = []
    for draw in range(N_DRAWS):
        torch.manual_seed(draw)
        circuit = build_skolik_qlayer(N_QUBITS, n_layers, ent=ent)
        with torch.no_grad():
            for p in circuit.parameters():
                p.copy_(torch.rand_like(p) * 2 * np.pi)
        per_qubit = [
            _relative_residual(
                _sample_sweep(circuit, q, SEED_OFFSET + draw * 10 + q), X, k_max
            )
            for q in range(N_QUBITS)
        ]
        out.append(float(np.mean(per_qubit)))
    return out


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_output_matches_the_full_L_frequency_fourier_basis(n_layers):
    residuals = _per_draw_mean_residuals(n_layers, ent=False, k_max=n_layers)
    assert max(residuals) < POSITIVE_TOL, (
        f"L={n_layers}: worst-draw relative residual {max(residuals):.2e} "
        f">= {POSITIVE_TOL:.0e} over {N_DRAWS} draws - the circuit is not the "
        f"degree-L trig polynomial the ceiling assumes (per-draw values: {residuals})"
    )


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_truncating_the_top_frequency_leaves_a_clear_residual(n_layers):
    """Negative control 1: without this, a small residual above proves
    nothing, since an over-flexible basis fits almost any bounded function.
    Assert on the MINIMUM over N_DRAWS set seeds, not one - see the module
    docstring on why a single draw can land in the tail by chance."""
    residuals = _per_draw_mean_residuals(n_layers, ent=False, k_max=n_layers - 1)
    assert min(residuals) > TRUNCATED_FLOOR, (
        f"L={n_layers}: best-draw (weakest) relative residual "
        f"{min(residuals):.2e} <= {TRUNCATED_FLOOR:.0e} over {N_DRAWS} draws - "
        f"the circuit may not reliably need its top frequency "
        f"(per-draw values: {residuals})"
    )


@pytest.mark.parametrize("n_layers", [2, 5])
def test_negative_control_entangled_circuit_is_not_a_clean_fourier_series(n_layers):
    """Negative control 2, and the one the sampling-design bug above weakened.
    n_layers=1 is excluded: the final-ring property makes `ent` a no-op at depth 1 on this
    template (the final layer's CZ ring cannot affect a PauliZ readout), so
    there would be nothing here to detect."""
    residuals = _per_draw_mean_residuals(n_layers, ent=True, k_max=n_layers)
    assert min(residuals) > ENTANGLED_FLOOR, (
        f"L={n_layers}: best-draw (weakest) relative residual "
        f"{min(residuals):.2e} <= {ENTANGLED_FLOOR:.0e} over {N_DRAWS} draws - "
        f"the entangled circuit fit the unentangled basis too well, the "
        f"negative control has no power (per-draw values: {residuals})"
    )


def test_unentangled_circuit_is_unaffected_by_the_other_wires_being_random():
    """Direct check of the emulator's product-state claim, using the SAME sampling
    design as the tests above: for ent=False, randomising the other wires
    must not matter at all - confirms the design change above isolates
    entanglement specifically, rather than just making every test harder."""
    residuals = _per_draw_mean_residuals(5, ent=False, k_max=5)
    assert max(residuals) < POSITIVE_TOL, residuals
