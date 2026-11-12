"""`SU2SkolikEmulator` must reproduce `build_skolik_qlayer(ent=False)`.

Why this needs a test rather than an argument
----------------------------------------------
`core/su2_emulator.py` claims that the unentangled `skolik` circuit is exactly
a product of independent single-qubit rotations, and represents each qubit as
a real Bloch vector instead of a complex amplitude. That claim is either
checkable or it is not science - so it is checked directly against the real
PennyLane `TorchLayer`, on the same weights, on every configuration the
emulator is meant to stand in for: forward output AND gradients, to ~1e-6.

The negative control matters as much as the positive ones. Comparing the
emulator (which never applies a CZ gate) against a REAL `ent=True` circuit
must produce a large discrepancy - if it didn't, the positive checks above
would not be testing anything, because a test that cannot fail proves
nothing. See `tests/test_entanglement_noop.py` for the sibling fact this
depends on: `ent=True` legitimately changes the circuit whenever
`n_layers > 1`, so `n_layers=5` in the negative control below is guaranteed
to differ from the emulator for a reason unrelated to the final-ring property.

Claim discipline (see core/su2_emulator.py's module docstring for the full
version): this test proves per-call agreement of forward output and
gradients. It says nothing about bitwise-identical training curves over a
full run - epsilon-greedy ties and buffer sampling amplify last-bit
differences over tens of thousands of steps. Do not cite this file for a
stronger claim than it makes.
"""
from __future__ import annotations

import time

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

from quantumrl_dissection.simulators.emulator import SU2SkolikEmulator  # noqa: E402

TOL = 1e-6

# (label, n_qubits, n_layers, transform_fn, input_sampler)
# input_sampler(batch) -> raw observation tensor, matching what the real arm feeds
# the circuit (before transform_fn - the emulator applies transform_fn itself,
# exactly as build_skolik_qlayer does).
CASES = [
    (
        "skolik_8q_cartpole_L5",
        8, 5, CartPoleNormalizationTransformer(),
        lambda b: torch.rand(b, 4) * 2 - 1,
    ),
    (
        "frozenlake_binary_depth1",
        4, 1, FrozenBasisToAngleTransformer("4x4"),
        lambda b: torch.randint(0, 16, (b, 1)).float(),
    ),
    (
        "frozenlake_binary_depth5",
        4, 5, FrozenBasisToAngleTransformer("4x4"),
        lambda b: torch.randint(0, 16, (b, 1)).float(),
    ),
    (
        "frozenlake_scalar_depth5",
        1, 5, FrozenNormalizationTransformer("4x4"),
        lambda b: torch.randint(0, 16, (b, 1)).float(),
    ),
]


def _paired(label, n_qubits, n_layers, transform_fn, sampler, seed=0):
    torch.manual_seed(seed)
    real = build_skolik_qlayer(n_qubits, n_layers, transform_fn=transform_fn, ent=False)
    emu = SU2SkolikEmulator(n_qubits, n_layers, transform_fn=transform_fn)
    emu.load_weights_from_torchlayer(real)
    x = sampler(16)
    return real, emu, x


@pytest.mark.parametrize("label,n_qubits,n_layers,transform_fn,sampler", CASES)
def test_forward_agrees(label, n_qubits, n_layers, transform_fn, sampler):
    real, emu, x = _paired(label, n_qubits, n_layers, transform_fn, sampler)
    with torch.no_grad():
        diff = (real(x) - emu(x)).abs().max().item()
    assert diff < TOL, f"{label}: forward disagreement {diff:.2e} >= {TOL:.0e}"


@pytest.mark.parametrize("label,n_qubits,n_layers,transform_fn,sampler", CASES)
def test_gradient_wrt_weights_agrees(label, n_qubits, n_layers, transform_fn, sampler):
    """`.mean()`, not `.sum()`, over the batch: summing 16 samples' gradients
    inflates float32 rounding error by the same factor without testing
    anything different - the claim is per-sample gradient agreement, and
    mean divides the compounded error back down to what a single sample
    would show."""
    real, emu, x = _paired(label, n_qubits, n_layers, transform_fn, sampler)
    real(x).mean().backward()
    emu(x).mean().backward()
    g_real = next(real.parameters()).grad
    g_emu = emu.weights.grad
    diff = (g_real - g_emu).abs().max().item()
    assert diff < TOL, f"{label}: gradient disagreement {diff:.2e} >= {TOL:.0e}"


def test_explicit_emb_indices_agrees():
    """The cycling path (n_data < n_wires) is covered by CASES above via the
    CartPole 8-qubit case; this covers the other branch - a caller-supplied
    `emb_indices` selecting a subset, no cycling needed."""
    torch.manual_seed(0)
    real = build_skolik_qlayer(3, 2, emb_indices=[1, 2, 3], ent=False)
    emu = SU2SkolikEmulator(3, 2, emb_indices=[1, 2, 3])
    emu.load_weights_from_torchlayer(real)
    x = torch.rand(16, 4) * 2 - 1
    with torch.no_grad():
        diff = (real(x) - emu(x)).abs().max().item()
    assert diff < TOL, f"explicit indices: forward disagreement {diff:.2e}"


@pytest.mark.parametrize("label,n_qubits,n_layers,transform_fn,sampler", CASES)
def test_negative_control_entangled_circuit_disagrees(label, n_qubits, n_layers, transform_fn, sampler):
    """The emulator never applies CZ. Compared against a REAL ent=True circuit
    on the same weights, it must differ substantially - otherwise the checks
    above are not exercising anything. n_layers=5 (or 1, see below) both work:
    the final-ring property says ent is a no-op only at n_layers_q=1 on THIS template, so the
    L1 case needs a depth bump to get a meaningful negative control."""
    if n_qubits == 1:
        pytest.skip(
            "ent=True is invalid at one qubit: the circular CZ becomes "
            "CZ(wires=[0, 0]), a self-loop PennyLane rejects. frozen_scalar_1q "
            "is necessarily unentangled by construction (see core/configs.py), "
            "so there is no entangled circuit to contrast against here."
        )
    depth = max(n_layers, 2)  # avoid the the final-ring property no-op at depth 1
    torch.manual_seed(0)
    ent_on = build_skolik_qlayer(n_qubits, depth, transform_fn=transform_fn, ent=True)
    emu = SU2SkolikEmulator(n_qubits, depth, transform_fn=transform_fn)
    with torch.no_grad():
        for p_on, p_emu in zip(ent_on.parameters(), [emu.weights]):
            p_emu.copy_(p_on)
    x = sampler(16)
    with torch.no_grad():
        diff = (ent_on(x) - emu(x)).abs().max().item()
    assert diff > 1e-3, (
        f"{label}: emulator agreed with an ENTANGLED circuit ({diff:.2e}) - "
        "the negative control has no power, something is wrong with the test"
    )


def test_throughput_reported():
    """Not a correctness assertion - the brief asks that throughput be
    measured and recorded, not asserted against a threshold (hardware varies
    several-fold across this project's own machines; see docs/reproducibility.md)."""
    n_qubits, n_layers, batch, reps = 8, 5, 32, 20
    transform_fn = CartPoleNormalizationTransformer()
    real = build_skolik_qlayer(n_qubits, n_layers, transform_fn=transform_fn, ent=False)
    emu = SU2SkolikEmulator(n_qubits, n_layers, transform_fn=transform_fn)
    emu.load_weights_from_torchlayer(real)
    x = torch.rand(batch, 4) * 2 - 1

    with torch.no_grad():
        real(x)  # warm up (lazy device / graph construction)
        t0 = time.perf_counter()
        for _ in range(reps):
            real(x)
        t_real = (time.perf_counter() - t0) / reps

        emu(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            emu(x)
        t_emu = (time.perf_counter() - t0) / reps

    print(
        f"\nskolik_8q_L5, batch={batch}: "
        f"PennyLane {1000*t_real:.2f} ms/call  |  SU2 emulator {1000*t_emu:.2f} ms/call  "
        f"|  speedup {t_real / max(t_emu, 1e-9):.1f}x  (measured on this machine; "
        f"see docs/reproducibility.md on not trusting a figure from elsewhere)"
    )
    assert t_emu > 0  # the only real assertion: it ran and produced a time


# ---------------------------------------------------------------------------
# Initial weights must come from the same distribution as the real layer.
#
# Everything above compares emulator and TorchLayer weight-for-weight, so it
# cannot see the initialisation. The emulator used to start at exactly zero
# while qml.qnn.TorchLayer starts uniformly in [0, 2*pi); a freshly built
# `su2_cartpole_L5` then trained as a different problem (5/10 seeds stuck at the
# CartPole floor against 0/9 for the real unentangled circuit). Without this
# test the "equivalence in distribution over seeds" claim in the module
# docstring would be unchecked.
# ---------------------------------------------------------------------------
def test_initial_weights_follow_the_real_layer_distribution():
    n_qubits, n_layers = 8, 50          # 800 weights: enough for a stable std
    torch.manual_seed(0)
    real = build_skolik_qlayer(n_qubits, n_layers, ent=False).weights.detach()
    torch.manual_seed(1)
    emu = SU2SkolikEmulator(n_qubits, n_layers).weights.detach()

    assert tuple(emu.shape) == tuple(real.shape)
    for name, w in (("real", real), ("emulator", emu)):
        assert float(w.min()) >= 0.0 and float(w.max()) <= 2 * np.pi + 1e-6, name
    # Uniform(0, 2*pi): mean = pi, std = 2*pi/sqrt(12) ~ 1.814.
    for name, w in (("real", real), ("emulator", emu)):
        assert abs(float(w.mean()) - np.pi) < 0.2, (name, float(w.mean()))
        assert abs(float(w.std()) - 2 * np.pi / np.sqrt(12)) < 0.15, (name, float(w.std()))
    assert float(emu.abs().max()) > 0.0, "emulator weights all zero again"
