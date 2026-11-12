"""
The additive Fourier ceiling - a classical control fixing the exact
representational limit of an UNENTANGLED, one-feature-per-wire circuit.

The argument, in one paragraph
-------------------------------
Per Schuld, Sweke & Meyer (2021): a data-reuploading circuit is, as a function
of one input feature, a truncated Fourier series - the ENCODING gate sets
which frequencies are accessible (here, integers 1..L for L reuploads of a
single-generator RX rotation), and the VARIATIONAL layers only set the
Fourier coefficients. `core/su2_emulator.py` establishes that
`build_skolik_qlayer(ent=False)` is exactly `n_qubits` independent single-wire
circuits (see that module's docstring for why). Combined: on a circuit whose
embedding puts one feature per wire (`angle_embedding`, not
`multiple_rotation_embedding` - see `check_additive_embedding` below), the
hypothesis class the UNENTANGLED circuit can express, per wire, at
reuploading depth L, is EXACTLY

    f(z) = w_0 + sum_{k=1}^{L} [ a_k cos(k z) + b_k sin(k z) ]

`FourierAdditiveCeiling` builds that basis directly: no PQC, no PennyLane, a
linear head on top of `{cos(kz), sin(kz)}` features (the bias `w_0` comes
from `nn.Linear`'s own bias term, so it is not built separately). Any
unentangled circuit's accessible function class is a SUBSET of what this head
can represent (same frequencies, and its coefficients are free real numbers
where the circuit's are constrained by unitarity - see the claim-discipline
note in `core/su2_emulator.py` and `docs/adjustments.md`), so if the
circuit ever beats this ceiling on held-out performance, something is wrong
with the comparison, not with the theory - it means the two are not
controlling the same thing (different information, different budget, or the
circuit is somehow also entangled).

Not applicable everywhere - guarded, not assumed
--------------------------------------------------
The derivation needs "one feature (or a constant, wire-independent function of
one feature) per wire". `hsiao` (`emb_type="multi"`) and `dr` whenever
`n_qubits < n_data` (verified against `simplyqrl/qlayers.py::build_dr_qlayer`:
the branch condition is literally `n_qubits < n_data`, so this is NOT limited
to 1-qubit Salinas - the 2-qubit Salinas arm already registered in
`core/configs.py` triggers the same non-additive path) both route through
`embeddings.multiple_rotation_embedding`, which composes THREE features on
EVERY wire via non-commuting Z-Y-Z rotations - not a sum of single-feature
functions. `check_additive_embedding` raises rather than silently building a
ceiling that does not bound anything there.

The FrozenLake Config B degeneracy
-----------------------------------
`FrozenBasisToAngleTransformer` maps each bit to {0, pi}. On that two-point
domain: `sin(k*0) = sin(k*pi) = 0` for every integer k - the sine features
are identically zero, carrying no information regardless of L. `cos(k*0) = 1`
always; `cos(k*pi) = (-1)^k` - so odd-k cosines all equal `1 - 2*b` (an affine
function of the bit) and every even-k cosine is the constant 1 (redundant
with the head's own bias). The entire 2L-dimensional feature set per wire
collapses to ONE informative direction, independent of L - exactly
`stats.iqm`-style degeneracy, but analytic rather than empirical, and it is
`docs/adjustments.md`'s pre-registered prediction P2 (depth cannot
rescue Config B without entanglement).

`linear_on_bits_ceiling` builds that degenerate case DIRECTLY - 5 parameters
per action (4 bits + bias), not the general 2*L*n_qubits+1 machinery carrying
mostly-zero and mostly-redundant columns. It is the same hypothesis class as
`FourierAdditiveCeiling` at any L on this domain; building it separately
avoids a rank-deficient linear head and states the L-independence in the
architecture itself rather than leaving it to be discovered by an optimiser.
`tests/test_frozenlake_additive_ceiling.py` proves the general claim (the
REAL circuit's output is affine in the bits, at every tested L) directly
against `build_skolik_qlayer`, independent of which module ships it.

Open sizing question, not resolved here
------------------------------------------
the matching recipe (`core/capacity.py::match_hidden_width`) solves for
a hidden width that spends AT LEAST the reference arm's measured parameter
budget, because that arm has a free capacity knob. This ceiling has no such
knob: `2*n_qubits*n_layers + n_actions` (or 5 per action, constant, on
FrozenLake Config B) is not a design choice, it is the exact size of the
accessible hypothesis class - inflating it would stop the comparison from
being a ceiling on THIS circuit's expressivity. Flagged for confirmation
rather than decided silently; see docs/design/EXPERIMENT-13-OR-VS-LR.md.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "check_additive_embedding",
    "FourierAdditiveCeiling",
    "linear_on_bits_ceiling",
]


def check_additive_embedding(
    circ_type: str, n_qubits: int, n_data: int
) -> None:
    """Raise unless this configuration embeds one feature (or a wire-shared
    function of one feature) per wire - the condition the Fourier ceiling's
    derivation needs. `n_data` is the RAW observation width, matching how
    `build_dr_qlayer` computes it (`inputs.shape[-1]`, before any index
    selection) - see docs/adjustments.md for why this is not just the
    1-qubit case.
    """
    if circ_type == "hsiao":
        raise ValueError(
            "circ_type='hsiao' (emb_type='multi') embeds three features on "
            "EVERY wire via a non-commuting Z-Y-Z composition "
            "(embeddings.multiple_rotation_embedding) - not additive across "
            "features. The Fourier ceiling's derivation does not apply "
            "without redefining its basis. See docs/adjustments.md."
        )
    if circ_type == "dr" and n_qubits < n_data:
        raise ValueError(
            f"circ_type='dr' with n_qubits ({n_qubits}) < n_data ({n_data}) "
            "routes through multiple_rotation_embedding (build_dr_qlayer's "
            "own branch condition), the same non-additive path as hsiao. "
            "This includes BOTH the 1-qubit and 2-qubit Salinas/dr arms "
            "currently registered in core/configs.py. "
            "See docs/adjustments.md."
        )
    if circ_type not in ("skolik", "dr", "basic"):
        raise ValueError(
            f"unknown circ_type {circ_type!r}: check_additive_embedding does "
            "not know whether this embedding is additive. Extend this guard "
            "before using the Fourier ceiling against it."
        )


class FourierAdditiveCeiling(nn.Module):
    """Classical hypothesis-class ceiling for an unentangled, one-feature-per-
    wire circuit at reuploading depth `n_layers`.

    Mirrors `SU2SkolikEmulator`'s constructor shape (`n_qubits`, `emb_indices`,
    `transform_fn`) deliberately, and reuses its exact feature-selection rule
    (same `transform_fn`, same indices-or-cycling logic), so the two can be
    pointed at the same arm's parameters without re-deriving which features
    land on which wire.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        n_actions: int,
        emb_indices: Optional[Sequence[int]] = None,
        transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        circ_type: str = "skolik",
        n_data: Optional[int] = None,
    ):
        super().__init__()
        if n_data is not None:
            check_additive_embedding(circ_type, n_qubits, n_data)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.emb_indices = list(emb_indices) if emb_indices is not None else None
        self.transform_fn = transform_fn
        # 2*L features per wire (cos, sin at k=1..L); the head's own bias
        # supplies the k=0 constant term, so it is not built separately.
        self.head = nn.Linear(2 * self.n_layers * self.n_qubits, n_actions)

    def _embed(self, data: torch.Tensor) -> torch.Tensor:
        """Identical to SU2SkolikEmulator._embed - see that module for the
        line-by-line correspondence with embeddings.angle_embedding."""
        if self.transform_fn is not None:
            data = self.transform_fn(data)
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)

        n_wires = self.n_qubits
        n_data = data.shape[-1]

        if self.emb_indices is not None:
            if len(self.emb_indices) != n_wires:
                raise ValueError(
                    f"emb_indices length ({len(self.emb_indices)}) must equal "
                    f"n_qubits ({n_wires})"
                )
            data = data[..., self.emb_indices]
            n_data = n_wires
        elif n_data > n_wires:
            data = data[..., :n_wires]

        if n_data < n_wires:
            idx = torch.arange(n_wires, device=data.device) % n_data
            data = data[..., idx]

        return data

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        squeeze = inputs.dim() == 1
        if squeeze:
            inputs = inputs.unsqueeze(0)

        z = self._embed(inputs)  # (B, n_qubits)
        k = torch.arange(1, self.n_layers + 1, dtype=z.dtype, device=z.device)  # (L,)
        angles = z.unsqueeze(-1) * k.view(1, 1, -1)  # (B, n_qubits, L)
        feats = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, n_qubits, 2L)
        feats = feats.reshape(feats.shape[0], -1)  # (B, n_qubits * 2L)

        out = self.head(feats)
        return out.squeeze(0) if squeeze else out


class _TransformedLinear(nn.Module):
    """`nn.Linear` with `transform_fn` applied to the raw input first.

    Exists so `linear_on_bits_ceiling` can be wired directly into the same
    env-observation plumbing every other arm receives (`build_arm_config`
    resolves a stored transform marker to a real callable before this ever
    runs - see `core/configs.py::_resolve_transform`), without baking a
    specific grid or transform into the ceiling module itself.
    """

    def __init__(self, transform_fn, linear: nn.Linear):
        super().__init__()
        self.transform_fn = transform_fn
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_fn is not None:
            x = self.transform_fn(x)
        return self.linear(x)


def linear_on_bits_ceiling(
    n_qubits: int, n_actions: int, transform_fn: Optional[Callable] = None
) -> nn.Module:
    """The FrozenLake-Config-B-degenerate Fourier ceiling: an affine model on
    the RAW bits, independent of reuploading depth L.

    Deliberately takes bits (0/1), not angles - `FrozenBasisToAngleTransformer`
    maps a bit to {0, pi} purely so the PQC can act on it; once the target
    function is known to be affine in the bit (this module's own docstring
    derives it, and `tests/test_frozenlake_additive_ceiling.py` verifies it
    against the real circuit), there is no reason to route through cos/sin at
    all. 5 parameters per action (4 weights + `nn.Linear`'s bias) x n_actions,
    matching the derivation in `docs/adjustments.md` exactly - not an
    approximation, not capacity-matched to anything, the exact size of the
    hypothesis class.

    `transform_fn`, if given, is applied to the raw observation before the
    linear head - e.g. `FrozenBasisToAngleTransformer` itself, which returns
    bits scaled by pi ({0, pi}) rather than {0, 1}. That constant rescaling does
    not change the hypothesis class (an affine model absorbs it into its
    weights), so it is fine to reuse the same transformer the real circuit
    uses rather than writing a separate bits-only one. Omit `transform_fn` to
    get the bare `nn.Linear` for callers that already have a bit vector
    in hand (e.g. the enumeration test in
    `tests/test_frozenlake_additive_ceiling.py`, which never constructs this
    function at all - it checks the claim directly against the real circuit).
    """
    linear = nn.Linear(n_qubits, n_actions)
    if transform_fn is None:
        return linear
    return _TransformedLinear(transform_fn, linear)
