"""
Batched torch statevector simulator of the `skolik` circuit, entangled or not.

What this is
------------
`build_skolik_qlayer` (src/simplyqrl/qlayers.py) is a PennyLane QNode wrapped in a
`qml.qnn.TorchLayer`. PennyLane's `lightning.qubit` IS a statevector simulator; the
physics is not what is slow here. What costs time at these sizes (4-8 qubits, i.e.
16-256 amplitudes) is the machinery around it: the TorchLayer evaluates the QNode
sample by sample, rebuilding and dispatching the circuit through Python for each one,
and differentiates with the adjoint method one sample at a time.

This module does the same computation as ONE batched tensor program: the state of the
whole batch is a complex tensor of shape (B, 2, ..., 2); every gate is a small einsum
over one qubit axis, a CZ is a multiplication by a precomputed +-1 mask, and the Pauli-Z
expectations are a matrix product with a sign table. Autograd differentiates through all
of it, on the same device as the rest of the network, with no per-sample Python loop.
It represents ENTANGLED circuits, which `SU2SkolikEmulator` cannot.

Same circuit, gate for gate
---------------------------
Per layer: RX(embedded feature) on every wire, then RY(weights[l, 0, i]) and
RZ(weights[l, 1, i]) on wire i, then - if `ent` - the circular CZ ring
`CZ(i, (i + 1) % n)` for i = 0..n-1, applied literally (so at n = 2 the ring applies
CZ(0, 1) and CZ(1, 0), which cancel, exactly as in the real circuit). After the last
layer, <Z_i> on every wire. The embedding (transform_fn, emb_indices, cycling) is
inherited from `SU2SkolikEmulator._embed`, and the weights have the same layout
(n_layers, 2, n_qubits) and the same initialisation (U[0, 2*pi)), so a trained
`TorchLayer`'s weights can be copied in with `load_weights_from_torchlayer`.

Claim discipline
----------------
`tests/test_statevector_equivalence.py` checks forward output and gradients against the
real PennyLane `TorchLayer`, entangled and not, on the same weights. As for the SU(2)
emulator, that is per-call agreement, not bitwise-identical training curves; the
training-level claim is equivalence in distribution over seeds, and it needs matching
initialisation. Internal arithmetic is complex128; the output is cast back to
the input dtype.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from quantumrl_dissection.simulators.emulator import SU2HybridAgent, SU2SkolikEmulator


def _apply_1q(psi: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Apply a 2x2 gate to `qubit`. `gate` is (2, 2) (shared) or (B, 2, 2) (per sample)."""
    batch = psi.shape[0]
    left, right = 2 ** qubit, 2 ** (n_qubits - qubit - 1)
    p = psi.reshape(batch, left, 2, right)
    if gate.dim() == 2:
        out = torch.einsum("kl,balr->bakr", gate, p)
    else:
        out = torch.einsum("bkl,balr->bakr", gate, p)
    return out.reshape(psi.shape)


class SVSkolikLayer(SU2SkolikEmulator):
    """Exact statevector version of `build_skolik_qlayer(..., ent=ent)`."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        emb_indices: Optional[Sequence[int]] = None,
        transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ent: bool = False,
        init_std: Optional[float] = None,
        trainable_input_scale: bool = False,
    ):
        super().__init__(n_qubits, n_layers, emb_indices=emb_indices, transform_fn=transform_fn,
                         init_std=init_std, trainable_input_scale=trainable_input_scale)
        self.ent = bool(ent)
        if self.ent and self.n_qubits < 2:
            raise ValueError(
                "ent=True needs at least 2 qubits (the real circuit would apply CZ(0, 0))."
            )
        n = self.n_qubits
        # Basis index b in [0, 2^n) has bit of qubit q equal to (b >> (n - 1 - q)) & 1
        # (qubit 0 is the most significant axis of the state tensor).
        idx = torch.arange(2 ** n)
        bits = torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)], dim=1)   # (2^n, n)
        self.register_buffer("_zsign", (1 - 2 * bits).to(torch.float64), persistent=False)
        if self.ent:
            sign = torch.ones(2 ** n, dtype=torch.float64)
            for i in range(n):                     # literal ring, including n = 2
                j = (i + 1) % n
                sign = sign * (1 - 2 * (bits[:, i] & bits[:, j])).to(torch.float64)
            self.register_buffer("_cz_ring", sign, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: (n_features,) or (B, n_features). Returns <Z_i>, shape (B, n_qubits)."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        squeeze = inputs.dim() == 1
        if squeeze:
            inputs = inputs.unsqueeze(0)

        angles = self._embed(inputs).to(torch.float64)         # (B, n)
        batch, n = angles.shape[0], self.n_qubits
        cdt = torch.complex128

        psi = torch.zeros(batch, 2 ** n, dtype=cdt, device=angles.device)
        psi[:, 0] = 1.0
        psi = psi.reshape((batch,) + (2,) * n)

        def rx_matrices(a: torch.Tensor) -> torch.Tensor:
            # RX(theta) = [[c, -i s], [-i s, c]], per sample and per qubit -> (B, n, 2, 2)
            c, s = torch.cos(a / 2), torch.sin(a / 2)
            zero = torch.zeros_like(c)
            return torch.stack([
                torch.stack([torch.complex(c, zero), torch.complex(zero, -s)], dim=-1),
                torch.stack([torch.complex(zero, -s), torch.complex(c, zero)], dim=-1),
            ], dim=-2)

        rx = None if self.trainable_input_scale else rx_matrices(angles)
        w = self.weights.to(torch.float64)
        for layer in range(self.n_layers):
            if self.trainable_input_scale:
                rx = rx_matrices(angles * self.input_scale[layer].to(torch.float64))
            wy, wz = w[layer, 0], w[layer, 1]                  # (n,)
            cy, sy = torch.cos(wy / 2), torch.sin(wy / 2)
            ry = torch.stack([torch.stack([cy, -sy], dim=-1),
                              torch.stack([sy, cy], dim=-1)], dim=-2).to(cdt)       # (n, 2, 2)
            rz = torch.diag_embed(torch.stack([torch.exp(-0.5j * wz), torch.exp(0.5j * wz)], dim=-1))
            m = rz @ ry                                        # RY first, then RZ  (n, 2, 2)
            for q in range(n):
                gate = m[q].unsqueeze(0) @ rx[:, q]            # (B, 2, 2): (RZ RY) RX
                psi = _apply_1q(psi, gate, q, n)
            if self.ent:
                psi = (psi.reshape(batch, 2 ** n) * self._cz_ring).reshape((batch,) + (2,) * n)

        probs = (psi.reshape(batch, 2 ** n).abs() ** 2)        # (B, 2^n) float64
        z = probs @ self._zsign                                # (B, n)
        z = z.to(inputs.dtype if inputs.is_floating_point() else torch.float32)
        return z.squeeze(0) if squeeze else z


class SVHybridAgent(SU2HybridAgent):
    """`SU2HybridAgent` with the statevector layer: accepts ent=True."""

    _ALLOWS_ENT = True

    @staticmethod
    def _build_q_layer(n_qubits: int, n_layers_q: int, config: dict):
        return SVSkolikLayer(
            n_qubits,
            n_layers_q,
            emb_indices=config.get("emb_indices", None),
            transform_fn=config.get("transform_fn", None),
            ent=bool(config.get("ent", False)),
            init_std=config.get("q_init_std", None),
            trainable_input_scale=bool(config.get("trainable_input_scale", False)),
        )
