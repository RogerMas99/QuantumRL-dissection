"""Models: circuit configurations, classical controls, the classical twin, Fourier ceilings and the environment encodings."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from simplyqrl.agents import HybridAgent


HYBRID: Dict[str, Any] = {
    "circ_type": "skolik",
    "n_qubits": 8,
    "n_layers_q": 5,
    "ent": True,
    "net_arch": [4],
    "activation": nn.Identity,
}

REUSE_FACTORS = [4, 8, 16, 32]
DEPTHS = [1, 2, 5]


def hybrid_reuse_config(reuse_repetitions: int, ent: Optional[bool] = None) -> Dict[str, Any]:
    """The hybrid agent with output reuse of factor ``reuse_repetitions``; ``ent=False`` removes the entangling gates."""
    cfg = dict(HYBRID)
    cfg["reuse_repetitions"] = int(reuse_repetitions)
    if ent is not None:
        cfg["ent"] = bool(ent)
    return cfg


def hybrid_depth_config(n_layers_q: int, ent: Optional[bool] = None) -> Dict[str, Any]:
    """The hybrid agent with circuit depth (data reuploading) ``n_layers_q``."""
    cfg = dict(HYBRID)
    cfg["n_layers_q"] = int(n_layers_q)
    if ent is not None:
        cfg["ent"] = bool(ent)
    return cfg


def hybrid_depth_reuse_config(n_layers_q: int, reuse_repetitions: int) -> Dict[str, Any]:
    """Depth and output reuse together (the crossing that the reference study does not run)."""
    cfg = dict(HYBRID)
    cfg["n_layers_q"] = int(n_layers_q)
    cfg["reuse_repetitions"] = int(reuse_repetitions)
    return cfg


def porting_skolik_config(n_qubits: int, n_layers_q: int, ent: bool = True) -> Dict[str, Any]:
    """Skolik-derived template: no hidden layer, input transformed as in the reference study."""
    return {
        "circ_type": "skolik",
        "n_qubits": int(n_qubits),
        "n_layers_q": int(n_layers_q),
        "ent": bool(ent),
        "transform_fn": "cartpole_norm",
    }


def porting_salinas_config(n_qubits: int, n_layers_q: int) -> Dict[str, Any]:
    """Perez-Salinas-derived template."""
    return {
        "circ_type": "dr",
        "n_qubits": int(n_qubits),
        "emb_indices": [1, 2, 3],
        "n_layers_q": int(n_layers_q),
        "ent": True,
        "transform_fn": "cartpole_norm",
    }


def porting_hsiao_config(reuse_repetitions: int, ent: bool = False) -> Dict[str, Any]:
    """Hsiao-derived template with output reuse; unentangled unless ``ent``."""
    return {
        "circ_type": "hsiao",
        "n_qubits": 4,
        "emb_indices": [1, 2, 3],
        "reuse_repetitions": int(reuse_repetitions),
        "n_layers_q": 1,
        "ent": bool(ent),
        "transform_fn": "arctan",
    }


PORTING_DEPTHS = [1, 2, 5]
PORTING_REUSE = [4, 8, 16, 32]

FROZEN_GRID = "4x4"

TRANSFORMS = {
    "frozen_scalar": "FrozenNormalizationTransformer",
    "frozen_binary": "FrozenBasisToAngleTransformer",
    "cartpole_norm": "CartPoleNormalizationTransformer",
    "arctan": "ArctanTransformer",
}


def resolve_transform(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a ``transform_fn`` marker into the library's transformer object (no-op if it is already a callable)."""
    marker = cfg.get("transform_fn")
    if not isinstance(marker, str):
        return cfg
    if marker not in TRANSFORMS:
        raise KeyError(f"unknown transform marker {marker!r}. Known: {sorted(TRANSFORMS)}")
    import simplyqrl.transformations as _t
    klass = getattr(_t, TRANSFORMS[marker])
    cfg["transform_fn"] = klass(FROZEN_GRID) if marker.startswith("frozen_") else klass()
    return cfg


def frozenlake_scalar_config(n_layers_q: int) -> Dict[str, Any]:
    """One qubit, the discrete state mapped to a single angle (necessarily unentangled: CZ(0, 0) is invalid)."""
    return {
        "circ_type": "skolik",
        "n_qubits": 1,
        "n_layers_q": int(n_layers_q),
        "ent": False,
        "transform_fn": "frozen_scalar",
        "net_arch": [],
    }


def frozenlake_binary_config(n_layers_q: int, ent: bool = True) -> Dict[str, Any]:
    """Four qubits, the state as a binary basis vector."""
    return {
        "circ_type": "skolik",
        "n_qubits": 4,
        "n_layers_q": int(n_layers_q),
        "ent": bool(ent),
        "transform_fn": "frozen_binary",
        "net_arch": [],
    }


FROZENLAKE_SCALAR_DEPTHS = [1, 5, 10, 15]
FROZENLAKE_BINARY_DEPTHS = [1, 5]


def count_trainable(module: "nn.Module") -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_agent_for(
    agent_type: str,
    agent_config: Dict[str, Any],
    env_id: str = "CartPole-v1",
    is_qnet: bool = True,
) -> "nn.Module":
    """Construct an agent without training it."""
    import gymnasium as gym

    from simplyqrl.agents import build_agent

    select = (agent_config or {}).get("_env_select")
    if select is not None:
        env = make_selected_env(env_id, select)()
    else:
        env = gym.make(env_id)
    try:
        obs_shape = env.observation_space.shape
        n_actions = int(env.action_space.n)
        return build_agent(agent_type, obs_shape, n_actions, agent_config,
                           is_qnet=is_qnet)
    finally:
        env.close()


def pqc_parameter_budget(hybrid_config: Dict[str, Any],
                         env_id: str = "CartPole-v1") -> Dict[str, int]:
    """Split a hybrid agent's parameters into quantum and classical head."""
    agent = build_agent_for("hybrid", hybrid_config, env_id=env_id, is_qnet=True)
    total = count_trainable(agent)

    quantum = 0
    for name, param in agent.named_parameters():
        if not param.requires_grad:
            continue
        if "q_layer" in name or "qlayer" in name or "weights" in name.split(".")[-1]:
            quantum += param.numel()

    return {"total": total, "quantum": quantum, "classical_head": total - quantum}


def match_hidden_width(
    target_params: int,
    in_dim: int,
    out_dim: int,
    max_width: int = 4096,
) -> Tuple[int, int]:
    """Smallest hidden width whose block spends closest to `target_params`."""
    if target_params <= 0:
        raise ValueError("target_params must be positive")

    def cost(h: int) -> int:
        return in_dim * h + h + h * out_dim + out_dim

    best_h, best_err = 1, abs(cost(1) - target_params)
    for h in range(1, max_width + 1):
        err = abs(cost(h) - target_params)
        if err < best_err:
            best_h, best_err = h, err
        if cost(h) > target_params * 3:
            break
    return best_h, cost(best_h)


TWIN_HIDDEN = 6


class ClassicalTwinAgent(HybridAgent):
    """``HybridAgent`` whose quantum layer is replaced by a classical extractor: Linear, ReLU, Linear, and Tanh when bounded."""

    def __init__(self, obs_dim: int, act_dim: int, config: dict, is_qnet: bool = True):
        if not is_qnet:
            raise ValueError(
                "ClassicalTwinAgent only supports is_qnet=True (DQN Q-networks); no actor-critic construction exists."
            )
        cfg = dict(config)
        hidden = int(cfg.pop("twin_hidden", TWIN_HIDDEN))
        bounded = bool(cfg.pop("twin_bounded", True))
        if cfg.get("circ_type") != "skolik":
            raise ValueError("ClassicalTwinAgent expects the config of a hybrid_reuse{R} arm (circ_type='skolik'): "
                             "the head is defined by that config.")
        super().__init__(obs_dim, act_dim, cfg, is_qnet=True)

        net = getattr(self, "network", None)
        if not isinstance(net, nn.Sequential) or net[0] is not getattr(self, "qnet_q_layer", None):
            raise RuntimeError("ClassicalTwinAgent: upstream HybridAgent no longer puts the quantum layer first in "
                               "self.network.")
        layers = [nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.n_qubits)]
        if bounded:
            layers.append(nn.Tanh())
        self.twin_bounded = bounded
        net[0] = nn.Sequential(*layers)
        del self.qnet_q_layer

    @property
    def extractor(self) -> nn.Module:
        return self.network[0]

    @property
    def head(self) -> nn.Sequential:
        """Everything after the extractor: [OutputReuse(R) if R > 1] -> Linear -> activation -> Linear."""
        return self.network[1:]


def check_additive_embedding(
    circ_type: str, n_qubits: int, n_data: int
) -> None:
    """Raise unless the embedding puts one feature (or a function of one feature) on each wire."""
    if circ_type == "hsiao":
        raise ValueError(
            "circ_type='hsiao' (emb_type='multi') embeds three features on "
            "EVERY wire via a non-commuting Z-Y-Z composition "
            "(embeddings.multiple_rotation_embedding) - not additive across "
            "features. The Fourier ceiling's derivation does not apply "
            "without redefining its basis."
        )
    if circ_type == "dr" and n_qubits < n_data:
        raise ValueError(
            f"circ_type='dr' with n_qubits ({n_qubits}) < n_data ({n_data}) "
            "routes through multiple_rotation_embedding (build_dr_qlayer's "
            "own branch condition), the same non-additive path as hsiao. "
            "This includes BOTH the 1-qubit and 2-qubit Salinas/dr arms "
            "registered for the 1-qubit and 2-qubit Salinas arms."
        )
    if circ_type not in ("skolik", "dr", "basic"):
        raise ValueError(
            f"unknown circ_type {circ_type!r}: check_additive_embedding does "
            "not know whether this embedding is additive. Extend this guard "
            "before using the Fourier ceiling against it."
        )


class FourierAdditiveCeiling(nn.Module):
    """Classical ceiling of an unentangled circuit with one feature per wire: cos and sin features up to the circuit depth, then a linear head."""

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
        self.head = nn.Linear(2 * self.n_layers * self.n_qubits, n_actions)

    def _embed(self, data: torch.Tensor) -> torch.Tensor:
        """Same wire assignment as the SU(2) emulator."""
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

        z = self._embed(inputs)
        k = torch.arange(1, self.n_layers + 1, dtype=z.dtype, device=z.device)
        angles = z.unsqueeze(-1) * k.view(1, 1, -1)
        feats = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        feats = feats.reshape(feats.shape[0], -1)

        out = self.head(feats)
        return out.squeeze(0) if squeeze else out


class _TransformedLinear(nn.Module):
    """`nn.Linear` with `transform_fn` applied to the raw input first."""

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
    """Affine model on the raw bits (the ceiling of the unentangled binary circuit), after the optional transform."""
    linear = nn.Linear(n_qubits, n_actions)
    if transform_fn is None:
        return linear
    return _TransformedLinear(transform_fn, linear)


FROZEN_SCALAR_ID = "FrozenLake4x4Scalar-v0"
FROZEN_ONEHOT_ID = "FrozenLake4x4OneHot-v0"

FROZEN_MAX_EPISODE_STEPS = 100

_FROZEN_KWARGS = {"map_name": "4x4", "is_slippery": False}


class DiscreteToBoxObs(gym.ObservationWrapper):
    """Present `Discrete(n)` as `Box(shape=(1,), float32)`."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Discrete):
            raise TypeError(
                f"DiscreteToBoxObs expects a Discrete space, got {env.observation_space}"
            )
        self.n_states = int(env.observation_space.n)
        self.observation_space = spaces.Box(
            low=0.0, high=float(self.n_states - 1), shape=(1,), dtype=np.float32
        )

    def observation(self, obs):
        return np.array([obs], dtype=np.float32)


class OneHotObs(gym.ObservationWrapper):
    """Present `Discrete(n)` as a one-hot `Box(shape=(n,), float32)`."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Discrete):
            raise TypeError(
                f"OneHotObs expects a Discrete space, got {env.observation_space}"
            )
        self.n_states = int(env.observation_space.n)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_states,), dtype=np.float32
        )

    def observation(self, obs):
        vec = np.zeros(self.n_states, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec


def _frozen_scalar(**kwargs):
    return DiscreteToBoxObs(gym.make("FrozenLake-v1", **_FROZEN_KWARGS, **kwargs))


def _frozen_onehot(**kwargs):
    return OneHotObs(gym.make("FrozenLake-v1", **_FROZEN_KWARGS, **kwargs))


_REGISTERED = False


def register_environments(force: bool = False) -> list:
    """Register the adapted FrozenLake ids."""
    global _REGISTERED
    if _REGISTERED and not force:
        return [FROZEN_SCALAR_ID, FROZEN_ONEHOT_ID]

    existing = set(gym.registry.keys())
    for env_id, entry in ((FROZEN_SCALAR_ID, _frozen_scalar),
                          (FROZEN_ONEHOT_ID, _frozen_onehot)):
        if env_id in existing and not force:
            continue
        gym.register(id=env_id, entry_point=entry,
                     max_episode_steps=FROZEN_MAX_EPISODE_STEPS)

    _REGISTERED = True
    return [FROZEN_SCALAR_ID, FROZEN_ONEHOT_ID]


REFERENCE_COMPONENTS = (1, 2, 3)


class SelectObs(gym.ObservationWrapper):
    """Keep only ``indices`` of a 1-D Box observation."""

    def __init__(self, env, indices=REFERENCE_COMPONENTS):
        super().__init__(env)
        self.indices = list(indices)
        box = env.observation_space
        self.observation_space = spaces.Box(low=box.low[self.indices], high=box.high[self.indices], dtype=box.dtype)

    def observation(self, obs):
        return obs[self.indices]


def make_selected_env(env_id: str = "CartPole-v1", indices=REFERENCE_COMPONENTS):
    """A zero-argument factory of the environment with the selected components."""
    return lambda: SelectObs(gym.make(env_id), indices)


BITS = 4
N_UNIQUE_WIRES = 4


class StateBits(nn.Module):
    """Maps the state index of FrozenLake 4x4 to its 4 bits, least significant first."""

    def __init__(self, n_bits: int = BITS):
        super().__init__()
        self.n_bits = n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        s = x[:, 0].round().long()
        idx = torch.arange(self.n_bits, device=x.device).unsqueeze(0)
        return ((s.unsqueeze(1) >> idx) & 1).to(x.dtype)


class BitsMLPAgent(nn.Module):
    """``StateBits -> Linear(4, w) -> ReLU -> Linear(w, n_actions)``; Q-network only."""

    def __init__(self, obs_dim: int, n_actions: int, config: Dict[str, Any], is_qnet: bool = True):
        super().__init__()
        if not is_qnet:
            raise ValueError("bits_mlp is DQN-only")
        w = int(config.get("width", 6))
        self.is_qnet = True
        self.network = nn.Sequential(StateBits(), nn.Linear(BITS, w), nn.ReLU(), nn.Linear(w, n_actions))

    def forward(self, x):
        return self.network(x)
