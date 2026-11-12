"""
Exact classical emulator of the unentangled `skolik` circuit.

Why this is possible at all
----------------------------
`build_skolik_qlayer(..., ent=False)` (src/simplyqrl/qlayers.py) never applies
a two-qubit gate. Every gate it uses - the `AngleEmbedding` rotation, `RY`,
`RZ` - acts on a single wire, and the circuit starts in the product state
`|0>^n`. A product state acted on only by single-qubit unitaries STAYS a
product state: there is nothing for those gates to entangle. So each qubit's
reduced state is exactly a pure single-qubit state throughout, fully
described by its Bloch vector - a real 3-vector - and `<Z_i>` is just that
vector's z-component. No amplitudes, no complex numbers, no 2^n scaling: this
module represents the whole circuit as `n_qubits` independent Bloch vectors,
updated by real 3x3 rotations, in O(n_qubits * n_layers).

This is not an approximation and not a variational ansatz of our own - it is
the same computation `build_skolik_qlayer(ent=False)` performs, written in a
basis that does not need a quantum simulator. `tests/test_su2_equivalence.py`
is the evidence: forward output and gradients must agree with the real
PennyLane `TorchLayer`, on the same weights, to ~1e-6.

What this module is NOT
------------------------
It does not support `ent=True` - CZ gates entangle, and an entangled two-qubit
state is not describable by two independent Bloch vectors, so there is no
"add entanglement" flag to add here. That absence is deliberate and is what
makes the negative control in the equivalence test meaningful: compare this
emulator's output against a REAL entangled circuit and the two must differ,
because they are different computations.

It is infrastructure and verification, not an experimental arm. Numbers in
the main results tables come from the real PennyLane path; anything produced
by this module is labelled as such wherever it is reported. See
docs/adjustments.md.

What is proven, and what is not
--------------------------------
Proven: agreement of forward output and of gradients, per call, to numerical
tolerance. NOT claimed: bitwise-identical training curves over a full run.
Epsilon-greedy argmax ties and replay-buffer sampling amplify last-bit
differences over tens of thousands of steps, so two functionally-equivalent
Q-networks can still diverge in which actions get sampled and which
transitions get stored. The training-level claim this module supports is
equivalence IN DISTRIBUTION over seeds, not per-trajectory identity. Never
state it more strongly than that.

Weight and embedding conventions - kept identical on purpose
--------------------------------------------------------------
- Weight layout `(n_layers, 2, n_qubits)`, same as `build_skolik_qlayer`, so a
  trained `TorchLayer`'s weights can be copied in directly
  (`load_weights_from_torchlayer`).
- Initial weights are drawn uniformly in [0, 2*pi), the default of
  `qml.qnn.TorchLayer` (`init_method=uniform_(b=2*pi)`), which is what the
  real circuit starts from. They used to start at exactly ZERO, which made a
  freshly built emulator a different training problem from the real circuit
  even though its forward/gradient agree weight-for-weight: see docs/adjustments.md. The equivalence-in-distribution claim above only holds
  with matching initialisation.
- Feature selection replicates `embeddings.angle_embedding` exactly: apply
  `transform_fn` first: then, if `indices` is given, select those features (one
  per wire); otherwise take the leading features if there are more of them
  than wires; and CYCLE (`arange(n_wires) % n_data`) if there are fewer -
  the same rule that puts each of CartPole's 4 features on 2 of Skolik's 8
  wires.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn

__all__ = ["SU2SkolikEmulator", "SU2HybridAgent"]


def _apply_rx(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Bloch-vector rotation about the x-axis. v: (..., 3), theta: (...) broadcastable."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([x, c * y - s * z, s * y + c * z], dim=-1)


def _apply_ry(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([c * x + s * z, y, -s * x + c * z], dim=-1)


def _apply_rz(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)


class SU2SkolikEmulator(nn.Module):
    """Classical, exact emulator of `build_skolik_qlayer(..., ent=False)`.

    Parameters mirror the PennyLane builder deliberately, so a call site can
    swap one for the other without renaming anything:

        build_skolik_qlayer(n_qubits, n_layers, emb_indices=..., transform_fn=..., ent=False)
        SU2SkolikEmulator(n_qubits, n_layers, emb_indices=..., transform_fn=...)

    `ent` is not a parameter here - see the module docstring for why.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        emb_indices: Optional[Sequence[int]] = None,
        transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        init_std: Optional[float] = None,
        trainable_input_scale: bool = False,
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.emb_indices = list(emb_indices) if emb_indices is not None else None
        self.transform_fn = transform_fn
        # Same layout as build_skolik_qlayer's weight_shapes = {"weights": (n_layers, 2, n_qubits)}.
        # Default init matches the real layer: U[0, 2*pi). `init_std` is an
        # EXPLORATORY option that draws N(0, init_std) instead; it is never the
        # default, so every arm that does not ask for it is unchanged.
        w = torch.empty(self.n_layers, 2, self.n_qubits)
        if init_std is None:
            w.uniform_(0.0, 2.0 * math.pi)
        else:
            w.normal_(0.0, float(init_std))
        self.weights = nn.Parameter(w)
        # EXPLORATORY option: one trainable multiplier per (layer, qubit) on the
        # embedded angle, initialised at 1 (so it starts as the plain circuit). Off by default.
        self.trainable_input_scale = bool(trainable_input_scale)
        if self.trainable_input_scale:
            self.input_scale = nn.Parameter(torch.ones(self.n_layers, self.n_qubits))

    def _embed(self, data: torch.Tensor) -> torch.Tensor:
        """Reproduces embeddings.angle_embedding's feature selection exactly
        (everything up to the AngleEmbedding template call, which RX below
        replaces)."""
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
            if max(self.emb_indices) >= n_data:
                raise ValueError("an index in emb_indices is out of range for the data")
            data = data[..., self.emb_indices]
            n_data = n_wires
        elif n_data > n_wires:
            data = data[..., :n_wires]

        if n_data < n_wires:
            idx = torch.arange(n_wires, device=data.device) % n_data
            data = data[..., idx]

        return data

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: (n_features,) or (B, n_features). Returns <Z_i>, shape (B, n_qubits)."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        squeeze = inputs.dim() == 1
        if squeeze:
            inputs = inputs.unsqueeze(0)

        angles = self._embed(inputs)  # (B, n_qubits)
        batch = angles.shape[0]

        v = torch.zeros(batch, self.n_qubits, 3, dtype=angles.dtype, device=angles.device)
        v = v.clone()
        v[..., 2] = 1.0  # Bloch vector of |0> is (0, 0, 1)

        for layer in range(self.n_layers):
            a = angles * self.input_scale[layer] if self.trainable_input_scale else angles
            v = _apply_rx(v, a)                      # AngleEmbedding(rotation='X')
            v = _apply_ry(v, self.weights[layer, 0])  # RY(weights[layer, 0, i])
            v = _apply_rz(v, self.weights[layer, 1])  # RZ(weights[layer, 1, i])
            # No CZ step: this module only ever represents ent=False.

        z = v[..., 2]  # (B, n_qubits)
        return z.squeeze(0) if squeeze else z

    def load_weights_from_torchlayer(self, torch_layer: Any) -> None:
        """Copy weights from a `qml.qnn.TorchLayer` built by
        `build_skolik_qlayer(..., ent=False, ...)` with the same n_qubits/n_layers.

        Guarded: raises if the source's weight tensor is not the shape this
        emulator expects, rather than silently copying a mismatched tensor.
        """
        src = getattr(torch_layer, "weights", None)
        if src is None:
            raise AttributeError(
                "torch_layer has no 'weights' parameter - is this really a "
                "TorchLayer built by build_skolik_qlayer?"
            )
        if tuple(src.shape) != tuple(self.weights.shape):
            raise ValueError(
                f"weight shape mismatch: source {tuple(src.shape)} vs "
                f"emulator {tuple(self.weights.shape)}"
            )
        with torch.no_grad():
            self.weights.copy_(src.detach())


# ---------------------------------------------------------------------------
# SU2HybridAgent - a DQN Q-network built exactly like
# `simplyqrl.agents.HybridAgent(circ_type="skolik", ent=False, is_qnet=True)`,
# with `SU2SkolikEmulator` in place of the real PennyLane `TorchLayer`.
#
# Not a new architecture: reproduces HybridAgent's is_qnet construction line
# for line - [PQC layer] -> [OutputReuse if reuse_repetitions>1] ->
# [pi_arch Linear+activation]* -> [final Linear] -> [OutputScale if
# requested] - reusing upstream's own OutputReuse/OutputScale rather than
# reimplementing them, so a config built for `hybrid`/`frozenlake_binary_*`
# etc. can be pointed at "su2" unchanged and produce the same-shaped network,
# just without a quantum simulator underneath. Dispatched via
# `core/compat.py`'s agent_type extension; see docs/adjustments.md.
#
# Deliberately guarded rather than permissive on circ_type/ent: silently
# emulating an entangled circuit as unentangled would be exactly the kind of
# assumed-not-checked mistake this project exists to avoid.
# ---------------------------------------------------------------------------
class SU2HybridAgent(nn.Module):
    """Q-network agent whose PQC layer is `SU2SkolikEmulator`.

    Only DQN Q-networks are supported (``is_qnet=True``): no actor-critic
    construction exists here, because no arm needs one.

    Config keys read (same names/semantics as `HybridAgent`): ``net_arch``,
    ``activation``, ``n_qubits``, ``n_layers_q``, ``emb_indices``,
    ``transform_fn``, ``reuse_repetitions``, ``use_output_scaling``,
    ``output_scale_init``. ``circ_type`` must be ``"skolik"`` (or absent) and
    ``ent`` must be falsy (or absent) - anything else raises, rather than
    silently emulating a circuit this class cannot represent.
    """

    # Hooks for the statevector subclass (core/statevector.py), which reuses this
    # construction line for line but accepts ent=True.
    _ALLOWS_ENT = False

    @staticmethod
    def _build_q_layer(n_qubits: int, n_layers_q: int, config: dict):
        return SU2SkolikEmulator(
            n_qubits,
            n_layers_q,
            emb_indices=config.get("emb_indices", None),
            transform_fn=config.get("transform_fn", None),
            init_std=config.get("q_init_std", None),
            trainable_input_scale=bool(config.get("trainable_input_scale", False)),
        )

    def __init__(self, obs_dim: int, act_dim: int, config: dict, is_qnet: bool = True):
        super().__init__()
        if not is_qnet:
            raise ValueError(
                "SU2HybridAgent only supports is_qnet=True (DQN Q-networks); "
                "no actor-critic construction exists."
            )
        circ_type = config.get("circ_type", "skolik")
        if circ_type != "skolik":
            raise ValueError(
                f"SU2HybridAgent only emulates circ_type='skolik' "
                f"(SU2SkolikEmulator's only regime); got circ_type={circ_type!r}."
            )
        if config.get("ent", False) and not self._ALLOWS_ENT:
            raise ValueError(
                "SU2HybridAgent only emulates ent=False (unentangled skolik) - "
                "an entangled two-qubit state is not describable as independent "
                "Bloch vectors. See this module's docstring."
            )

        # Lazy: keeps this module importable without the quantum stack
        # (mirrors dqn/safe.py's own lazy-import discipline). OutputReuse and
        # OutputScale are reused from upstream rather than reimplemented.
        from simplyqrl.agents import OutputReuse, OutputScale

        net_arch = config.get("net_arch", {"pi": [], "vf": []})
        if isinstance(net_arch, dict):
            pi_arch = net_arch.get("pi", [])
        elif isinstance(net_arch, list):
            pi_arch = net_arch
        else:
            raise ValueError("net_arch must be either a dict or a list")

        activation = config.get("activation", nn.ReLU)
        n_qubits = int(config.get("n_qubits", 4))
        n_layers_q = config.get("n_layers_q", 1)
        self.n_qubits = n_qubits
        self.reuse_repetitions = config.get("reuse_repetitions", 1)

        self.q_layer = self._build_q_layer(n_qubits, n_layers_q, config)

        layers: list = [self.q_layer]
        if self.reuse_repetitions > 1:
            layers.append(OutputReuse(self.reuse_repetitions))
            prev_dim = n_qubits * self.reuse_repetitions
        else:
            prev_dim = n_qubits

        for size in pi_arch:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(activation())
            prev_dim = size
        layers.append(nn.Linear(prev_dim, act_dim))

        # Unlike upstream's HybridAgent, this is new code: appended
        # correctly the first time, no post-hoc patch needed.
        if config.get("use_output_scaling", False):
            init_val = config.get("output_scale_init", 2.0)
            layers.append(OutputScale(act_dim, init_val))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
