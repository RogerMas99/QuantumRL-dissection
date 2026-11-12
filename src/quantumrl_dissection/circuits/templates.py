"""Circuit templates and their configurations.

A configuration is the dictionary that the vendored library's ``HybridAgent`` receives (``circ_type``, ``n_qubits``,
``n_layers_q``, ``ent``, ``net_arch``, ``activation``, ``transform_fn``, ``reuse_repetitions``, ...). Nothing here builds a
circuit: the circuits themselves are the vendored library's (``third_party/simplyqrl/qlayers.py``) and the simulators'.

The hybrid agent of the study is the DQN example of the library (8 qubits, the Skolik-derived template at depth 5, an
entangling ring in every layer, a head with a linear intermediate layer of width four). The configurations of the
reference study are transcribed from its scripts without modification.

``transform_fn`` is stored as a marker string and resolved to the library's transformer object when an arm is built
(``resolve_transform``), so that this module imports without the library.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

# ---------------------------------------------------------------------------------------------------------------------
# The hybrid agent and its variations
# ---------------------------------------------------------------------------------------------------------------------
HYBRID: Dict[str, Any] = {
    "circ_type": "skolik",
    "n_qubits": 8,
    "n_layers_q": 5,
    "ent": True,
    "net_arch": [4],
    "activation": nn.Identity,
}

REUSE_FACTORS = [4, 8, 16, 32]      # output reuse factors R of the reference study
DEPTHS = [1, 2, 5]                  # data reuploading depths L (the hybrid agent has L = 5)


def hybrid_reuse_config(reuse_repetitions: int, ent: Optional[bool] = None) -> Dict[str, Any]:
    """The hybrid agent plus output reuse of factor ``reuse_repetitions`` (``ent=False`` removes the entangling ring)."""
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


# ---------------------------------------------------------------------------------------------------------------------
# Configurations of the reference study (transcribed from its scripts)
# ---------------------------------------------------------------------------------------------------------------------
def porting_skolik_config(n_qubits: int, n_layers_q: int, ent: bool = True) -> Dict[str, Any]:
    """Skolik-derived template: no hidden layer in the head, input transformed by the reference study's normalisation."""
    return {
        "circ_type": "skolik",
        "n_qubits": int(n_qubits),
        "n_layers_q": int(n_layers_q),
        "ent": bool(ent),
        "transform_fn": "cartpole_norm",
    }


def porting_salinas_config(n_qubits: int, n_layers_q: int) -> Dict[str, Any]:
    """Perez-Salinas-derived template. The embedding has three rotations, so the observation is reduced to [1, 2, 3]."""
    return {
        "circ_type": "dr",
        "n_qubits": int(n_qubits),
        "emb_indices": [1, 2, 3],
        "n_layers_q": int(n_layers_q),
        "ent": True,                  # no effect at one qubit
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

# ---------------------------------------------------------------------------------------------------------------------
# FrozenLake circuits
# ---------------------------------------------------------------------------------------------------------------------
FROZEN_GRID = "4x4"

TRANSFORMS = {
    "frozen_scalar": "FrozenNormalizationTransformer",    # state -> one angle
    "frozen_binary": "FrozenBasisToAngleTransformer",     # state -> 4 bits -> {0, pi}
    "cartpole_norm": "CartPoleNormalizationTransformer",  # Skolik- and Salinas-derived configurations
    "arctan": "ArctanTransformer",                        # Hsiao-derived configurations
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
    # The FrozenLake transformers need the grid size; the CartPole ones take no argument.
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
