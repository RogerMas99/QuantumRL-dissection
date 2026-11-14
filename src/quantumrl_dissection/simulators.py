"""Classical simulators of the circuits: an exact SU(2) emulator (unentangled circuits) and a batched state-vector simulator."""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn
from simplyqrl.agents import OutputReuse, OutputScale


def _apply_rx(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Bloch-vector rotation about the x-axis."""
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
    """Exact emulator of the unentangled Skolik circuit: each qubit is a Bloch vector rotated by RX (data), RY and RZ (weights)."""

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
        w = torch.empty(self.n_layers, 2, self.n_qubits)
        if init_std is None:
            w.uniform_(0.0, 2.0 * math.pi)
        else:
            w.normal_(0.0, float(init_std))
        self.weights = nn.Parameter(w)
        self.trainable_input_scale = bool(trainable_input_scale)
        if self.trainable_input_scale:
            self.input_scale = nn.Parameter(torch.ones(self.n_layers, self.n_qubits))

    def _embed(self, data: torch.Tensor) -> torch.Tensor:
        """Feature selection of the library's angle embedding, up to the rotations."""
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
        """inputs: (n_features,) or (B, n_features)."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        squeeze = inputs.dim() == 1
        if squeeze:
            inputs = inputs.unsqueeze(0)

        angles = self._embed(inputs)
        batch = angles.shape[0]

        v = torch.zeros(batch, self.n_qubits, 3, dtype=angles.dtype, device=angles.device)
        v = v.clone()
        v[..., 2] = 1.0

        for layer in range(self.n_layers):
            a = angles * self.input_scale[layer] if self.trainable_input_scale else angles
            v = _apply_rx(v, a)
            v = _apply_ry(v, self.weights[layer, 0])
            v = _apply_rz(v, self.weights[layer, 1])

        z = v[..., 2]
        return z.squeeze(0) if squeeze else z

    def load_weights_from_torchlayer(self, torch_layer: Any) -> None:
        """Copy the weights of a ``qml.qnn.TorchLayer`` built with the same configuration."""
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


class SU2HybridAgent(nn.Module):
    """Q-network agent whose PQC layer is `SU2SkolikEmulator`."""

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

        if config.get("use_output_scaling", False):
            init_val = config.get("output_scale_init", 2.0)
            layers.append(OutputScale(act_dim, init_val))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _apply_1q(psi: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Apply a 2x2 gate to `qubit`."""
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
        idx = torch.arange(2 ** n)
        bits = torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)], dim=1)
        self.register_buffer("_zsign", (1 - 2 * bits).to(torch.float64), persistent=False)
        if self.ent:
            sign = torch.ones(2 ** n, dtype=torch.float64)
            for i in range(n):
                j = (i + 1) % n
                sign = sign * (1 - 2 * (bits[:, i] & bits[:, j])).to(torch.float64)
            self.register_buffer("_cz_ring", sign, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: (n_features,) or (B, n_features)."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        squeeze = inputs.dim() == 1
        if squeeze:
            inputs = inputs.unsqueeze(0)

        angles = self._embed(inputs).to(torch.float64)
        batch, n = angles.shape[0], self.n_qubits
        cdt = torch.complex128

        psi = torch.zeros(batch, 2 ** n, dtype=cdt, device=angles.device)
        psi[:, 0] = 1.0
        psi = psi.reshape((batch,) + (2,) * n)

        def rx_matrices(a: torch.Tensor) -> torch.Tensor:
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
            wy, wz = w[layer, 0], w[layer, 1]
            cy, sy = torch.cos(wy / 2), torch.sin(wy / 2)
            ry = torch.stack([torch.stack([cy, -sy], dim=-1),
                              torch.stack([sy, cy], dim=-1)], dim=-2).to(cdt)
            rz = torch.diag_embed(torch.stack([torch.exp(-0.5j * wz), torch.exp(0.5j * wz)], dim=-1))
            m = rz @ ry
            for q in range(n):
                gate = m[q].unsqueeze(0) @ rx[:, q]
                psi = _apply_1q(psi, gate, q, n)
            if self.ent:
                psi = (psi.reshape(batch, 2 ** n) * self._cz_ring).reshape((batch,) + (2,) * n)

        probs = (psi.reshape(batch, 2 ** n).abs() ** 2)
        z = probs @ self._zsign
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
