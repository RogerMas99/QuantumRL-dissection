"""The registry of arms: one name -> (agent type, configuration).

The name of an arm describes its specification; the experiment that runs it is a property of the run.

    hybrid[_unentangled][_depth{L}][_reuse{R}]        the hybrid agent (depth 5 is the default and is omitted)
    reference_control[_reuse{R}]                      the classical control of the reference study (R = 4: no suffix)
    matched_control[_reuse{R}|_replicated{R}]         classical control matched in parameters (derived)
    classical_twin[_unbounded][_reuse{R}]             the hybrid with its circuit replaced by a classical extractor
    classical_twin_{collapsed{R},head_lr_x{k},grad_x{R},all_lr_x{k}}   learning-rate variants of the twin
    fourier_ceiling[_distinct[_lr_x2[_collapsed]]]    additive Fourier ceilings of the CartPole circuit
    network_no_replication_w{w}[_lr_x4], ...          classical networks without replication of the input
    large_network                                     the large network
    porting_{skolik,salinas,hsiao}_...                the configurations of the reference study run under DQN
    frozenlake_...                                    FrozenLake arms
    sim_{emulator,statevector}_<arm>                  runs on a simulator of the circuit
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import torch.nn as nn

from quantumrl_dissection.models import (
    DEPTHS, FROZEN_ONEHOT_ID, N_UNIQUE_WIRES, FROZEN_SCALAR_ID, FROZENLAKE_BINARY_DEPTHS, FROZENLAKE_SCALAR_DEPTHS, HYBRID,
    PORTING_DEPTHS, PORTING_REUSE, REUSE_FACTORS, frozenlake_binary_config, frozenlake_scalar_config, hybrid_depth_config,
    hybrid_depth_reuse_config, hybrid_reuse_config, match_hidden_width, porting_hsiao_config, porting_salinas_config,
    porting_skolik_config, pqc_parameter_budget, register_environments, resolve_transform,
)

# Learning-rate variants: reuse factors R and learning-rate factors k.
RS_COLLAPSED = (4, 8, 16, 32)
RS_GRAD = (8, 32)
KS_HEAD = (2, 4, 8, 16, 32, 64, 128)
KS_GLOBAL = (0.5, 2, 4, 8)


def kname(k) -> str:
    return str(k).replace(".", "p")


def classical_models_with_reuse(hybrid: Dict[str, Any], reuse_indices) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """Classical models that contain output reuse, and their counterparts without replication."""
    base = {"n_qubits": hybrid["n_qubits"], "n_layers_q": hybrid["n_layers_q"], "circ_type": "skolik", "lr_classical": True}
    uniq = dict(base, n_qubits=N_UNIQUE_WIRES)
    rep4 = {"reuse_indices": list(reuse_indices), "n_repeats": 4, "net_arch": [7], "activation": nn.ReLU,
            "lr_classical": True, "lr_classical_R": 4}
    norep = {"reuse_indices": list(reuse_indices), "n_repeats": 1, "activation": nn.ReLU, "lr_classical": True}
    bits = {"lr_classical": True}
    return {
        "fourier_ceiling_distinct": ("fourier_additive", dict(uniq)),
        "fourier_ceiling_distinct_lr_x2": ("fourier_additive", dict(uniq, lr_head_mult=2.0)),
        "fourier_ceiling_distinct_lr_x2_collapsed": ("fourier_additive", dict(base, lr_classical_R=2, lr_collapse_R=2, lr_head_mult=2.0)),
        "network_no_replication_w7_lr_x4": ("classic", dict(norep, net_arch=[7], lr_head_mult=4.0)),
        "network_no_replication_w7_lr_x4_collapsed": ("classic", dict(rep4, lr_collapse_R=4, lr_head_mult=4.0)),
        "network_no_replication_w18": ("classic", dict(norep, net_arch=[18])),
        "network_no_replication_w18_lr_x4": ("classic", dict(norep, net_arch=[18], lr_head_mult=4.0)),
        "frozenlake_bits_network_w6": ("bits_mlp", dict(bits, width=6)),
        "frozenlake_bits_network_w12": ("bits_mlp", dict(bits, width=12)),
    }


register_environments()

OBSERVATION_COMPONENTS = [1, 2, 3]
MATCHED_INDICES_FULL = [0, 1, 2, 3]
MATCHED_INDICES_REFERENCE = [1, 2, 3]
REFERENCE_REPEATS = 4


def reference_control_config(n_repeats: int) -> Dict[str, Any]:
    """Linear head on the observation components [1, 2, 3], replicated ``n_repeats`` times, no hidden layer."""
    if int(n_repeats) == 1:
        return {"n_repeats": 1, "net_arch": [], "_env_select": list(OBSERVATION_COMPONENTS)}
    return {"reuse_indices": list(OBSERVATION_COMPONENTS), "n_repeats": int(n_repeats), "net_arch": []}


LARGE_NETWORK: Dict[str, Any] = {"net_arch": [120, 84], "activation": nn.ReLU}

ARMS: Dict[str, Any] = {
    "reference_control": ("classic", reference_control_config(REFERENCE_REPEATS)),
    "hybrid": ("hybrid", HYBRID),
    "large_network": ("classic", LARGE_NETWORK),
}

for _R in (1, 8, 16, 32):
    ARMS[f"reference_control_reuse{_R}"] = ("classic", reference_control_config(_R))
for _R in REUSE_FACTORS:
    ARMS[f"hybrid_reuse{_R}"] = ("hybrid", hybrid_reuse_config(_R))
    ARMS[f"hybrid_unentangled_reuse{_R}"] = ("hybrid", hybrid_reuse_config(_R, ent=False))
ARMS["hybrid_unentangled"] = ("hybrid", dict(HYBRID, ent=False))
for _L in (1, 2):
    ARMS[f"hybrid_depth{_L}"] = ("hybrid", hybrid_depth_config(_L))
    for _R in REUSE_FACTORS:
        ARMS[f"hybrid_depth{_L}_reuse{_R}"] = ("hybrid", hybrid_depth_reuse_config(_L, _R))
for _L in DEPTHS:
    if _L != 5:
        ARMS[f"hybrid_unentangled_depth{_L}"] = ("hybrid", hybrid_depth_config(_L, ent=False))

TWIN_REPEATS = [1, 4, 8, 16, 32]


def _twin(R: int, bounded: bool = True) -> Dict[str, Any]:
    return dict(hybrid_reuse_config(R), twin_hidden=6, twin_bounded=bounded)


def twin_name(R: int, unbounded: bool = False) -> str:
    base = "classical_twin_unbounded" if unbounded else "classical_twin"
    return base if int(R) == 1 else f"{base}_reuse{int(R)}"


for _R in TWIN_REPEATS:
    ARMS[twin_name(_R)] = ("classical_twin", _twin(_R, True))
    ARMS[twin_name(_R, unbounded=True)] = ("classical_twin", _twin(_R, False))

REFERENCE_STUDY_RESULTS: Dict[str, str] = {}
for _nq in (4, 8):
    for _L in PORTING_DEPTHS:
        _a = f"porting_skolik_{_nq}q_depth{_L}"
        ARMS[_a] = ("hybrid", porting_skolik_config(_nq, _L))
        REFERENCE_STUDY_RESULTS[_a] = f"Skolik_{_nq}Q_L{_L}"
for _nq in (1, 2):
    for _L in PORTING_DEPTHS:
        _a = f"porting_salinas_{_nq}q_depth{_L}"
        ARMS[_a] = ("hybrid", porting_salinas_config(_nq, _L))
        REFERENCE_STUDY_RESULTS[_a] = f"Salinas_{_nq}Q_L{_L}"
for _R in PORTING_REUSE:
    _a = f"porting_hsiao_reuse{_R}"
    ARMS[_a] = ("hybrid", porting_hsiao_config(_R))
    REFERENCE_STUDY_RESULTS[_a] = f"Quantum_r{_R}"
    REFERENCE_STUDY_RESULTS["reference_control" if _R == REFERENCE_REPEATS else f"reference_control_reuse{_R}"] = f"Classical_r{_R}"
for _L in (2, 5):
    _a = f"porting_skolik_4q_depth{_L}_unentangled"
    ARMS[_a] = ("hybrid", porting_skolik_config(4, _L, ent=False))
    REFERENCE_STUDY_RESULTS[_a] = f"Skolik_DR_L{_L}_Unentangled"
for _R in (4, 16):
    _a = f"porting_hsiao_reuse{_R}_entangled"
    ARMS[_a] = ("hybrid", porting_hsiao_config(_R, ent=True))
    REFERENCE_STUDY_RESULTS[_a] = f"Hsiao_OR_r{_R}_Entangled"

FROZENLAKE_ONEHOT_REFERENCE: Dict[str, Any] = {"net_arch": [64, 64], "activation": nn.ReLU}
FROZENLAKE_LARGE_NETWORK_SCALAR: Dict[str, Any] = {"net_arch": [64, 64], "activation": nn.ReLU}

for _L in FROZENLAKE_SCALAR_DEPTHS:
    ARMS[f"frozenlake_scalar_depth{_L}"] = ("hybrid", frozenlake_scalar_config(_L))
    ARMS[f"frozenlake_scalar_ceiling_depth{_L}"] = (
        "fourier_additive", {"n_qubits": 1, "n_layers_q": _L, "circ_type": "skolik", "transform_fn": "frozen_scalar"})
for _L in FROZENLAKE_BINARY_DEPTHS:
    ARMS[f"frozenlake_binary_depth{_L}"] = ("hybrid", frozenlake_binary_config(_L))
    ARMS[f"frozenlake_binary_unentangled_depth{_L}"] = ("hybrid", frozenlake_binary_config(_L, ent=False))
ARMS["frozenlake_onehot_reference"] = ("classic", FROZENLAKE_ONEHOT_REFERENCE)
ARMS["frozenlake_large_network_scalar"] = ("classic", FROZENLAKE_LARGE_NETWORK_SCALAR)
ARMS["frozenlake_binary_linear_ceiling"] = ("linear_on_bits", {"n_qubits": 4, "transform_fn": "frozen_binary"})

ARMS["fourier_ceiling"] = (
    "fourier_additive",
    {"n_qubits": HYBRID["n_qubits"], "n_layers_q": HYBRID["n_layers_q"], "circ_type": "skolik"},
)

SIM_REUSE = [1, 4, 8, 16, 32]


def sim_hybrid_name(kind: str, depth: int = 5, reuse: int = 1, unentangled: bool = False) -> str:
    s = f"sim_{kind}_hybrid"
    if unentangled:
        s += "_unentangled"
    if int(depth) != 5:
        s += f"_depth{int(depth)}"
    if int(reuse) != 1:
        s += f"_reuse{int(reuse)}"
    return s


def _cartpole_sim_cfg(depth: int, reuse: int, ent: bool) -> Dict[str, Any]:
    cfg = dict(HYBRID, n_layers_q=int(depth), ent=ent)
    if reuse > 1:
        cfg["reuse_repetitions"] = int(reuse)
    return cfg


for _R in SIM_REUSE:
    for _L in (1, 2, 5):
        ARMS[sim_hybrid_name("statevector", _L, _R)] = ("sv", _cartpole_sim_cfg(_L, _R, True))
    ARMS[sim_hybrid_name("statevector", 5, _R, unentangled=True)] = ("sv", _cartpole_sim_cfg(5, _R, False))
    ARMS[sim_hybrid_name("emulator", 5, _R, unentangled=True)] = ("su2", _cartpole_sim_cfg(5, _R, False))
for _L in (1, 2):
    ARMS[sim_hybrid_name("statevector", _L, 1, unentangled=True)] = ("sv", _cartpole_sim_cfg(_L, 1, False))
    ARMS[sim_hybrid_name("emulator", _L, 1, unentangled=True)] = ("su2", _cartpole_sim_cfg(_L, 1, False))
for _nq in (4, 8):
    for _L in (1, 2, 5):
        ARMS[f"sim_statevector_porting_skolik_{_nq}q_depth{_L}"] = ("sv", porting_skolik_config(_nq, _L, ent=True))
for _L in (2, 5):
    ARMS[f"sim_emulator_porting_skolik_4q_depth{_L}_unentangled"] = ("su2", porting_skolik_config(4, _L, ent=False))
for _L in FROZENLAKE_BINARY_DEPTHS:
    ARMS[f"sim_statevector_frozenlake_binary_depth{_L}"] = ("sv", frozenlake_binary_config(_L, ent=True))
    ARMS[f"sim_statevector_frozenlake_binary_unentangled_depth{_L}"] = ("sv", frozenlake_binary_config(_L, ent=False))
    ARMS[f"sim_emulator_frozenlake_binary_unentangled_depth{_L}"] = ("su2", frozenlake_binary_config(_L, ent=False))
for _L in (5, 10, 15):
    ARMS[f"sim_emulator_frozenlake_scalar_depth{_L}"] = ("su2", frozenlake_scalar_config(_L))


def _register_learning_rate_arms() -> None:
    for prefix in ("classical_twin", "sim_statevector_hybrid"):
        def base(R: int):
            if prefix == "classical_twin":
                return "classical_twin", _twin(R, True)
            return "sv", _cartpole_sim_cfg(5, R, True)
        for R in RS_COLLAPSED:
            at, cfg = base(R)
            ARMS[f"{prefix}_collapsed{R}"] = (at, dict(cfg, lr_collapse_R=R, lr_head_mult=float(R)))
        at, cfg1 = base(1)
        for R in RS_GRAD:
            ARMS[f"{prefix}_grad_x{R}"] = (at, dict(cfg1, lr_grad_mult=float(R)))
        for k in KS_HEAD:
            ARMS[f"{prefix}_head_lr_x{kname(k)}"] = (at, dict(cfg1, lr_head_mult=float(k)))
        for k in KS_GLOBAL:
            ARMS[f"{prefix}_all_lr_x{kname(k)}"] = (at, dict(cfg1, lr_global_mult=float(k)))


_register_learning_rate_arms()


def _register_classical_models_with_reuse() -> None:
    """Classical models that contain output reuse, and their counterparts without replication."""
    for name, entry in classical_models_with_reuse(HYBRID, MATCHED_INDICES_FULL).items():
        ARMS[name] = entry


_register_classical_models_with_reuse()

ENVIRONMENTS: Dict[str, Dict[str, Any]] = {
    "cartpole": {"env_id": "CartPole-v1", "obs_dim": 4, "n_actions": 2, "reference_components": OBSERVATION_COMPONENTS},
    "frozenlake": {
        "env_id": FROZEN_SCALAR_ID,
        "raw_env_id": "FrozenLake-v1",
        "env_kwargs": {"map_name": "4x4", "is_slippery": False},
        "obs_dim": 1,
        "n_actions": 4,
        "onehot_env_id": FROZEN_ONEHOT_ID,
        "max_episode_steps": 100,
    },
}


def environment_of(arm: str) -> str:
    """Registered environment id that an arm expects."""
    if "frozenlake" in arm:
        return FROZEN_ONEHOT_ID if arm == "frozenlake_onehot_reference" else FROZEN_SCALAR_ID
    return "CartPole-v1"


FROZENLAKE_MATCHED_REFERENCE = "frozenlake_scalar_depth5"
_MATCHED_REUSE = re.compile(r"^matched_control_reuse(\d+)$")
_MATCHED_REPLICATED = re.compile(r"^matched_control_replicated(\d+)$")


def _matched_cfg(budgets: Dict[str, int], match_to: str, n_repeats: int, activation, indices, matched_to: str, in_dim: int):
    budget = budgets["total"] if match_to == "total" else budgets["quantum"]
    width, spent = match_hidden_width(budget, in_dim=in_dim, out_dim=2)
    return {
        "reuse_indices": list(indices),
        "n_repeats": n_repeats,
        "net_arch": [width],
        "activation": activation,
        "_matched_to": matched_to,
        "_match_target": match_to,
        "_budget": budget,
        "_hybrid_total": budgets["total"],
        "_hybrid_quantum": budgets["quantum"],
        "_spent_params": spent,
    }


def build_arm_config(
    arm: str,
    hybrid_reference: str = "hybrid",
    activation: type = nn.ReLU,
    env_id: str = "CartPole-v1",
    match_to: str = "total",
    observation: str = "full",
    **overrides: Any,
):
    """Return (agent_type, agent_config) for an arm name."""

    if match_to not in ("total", "quantum"):
        raise ValueError(f"match_to must be 'total' or 'quantum', got {match_to!r}")

    if arm == "matched_control":
        ref_type, ref_cfg = ARMS[hybrid_reference]
        if ref_type != "hybrid":
            raise ValueError("hybrid_reference must name a hybrid arm")
        budgets = pqc_parameter_budget(ref_cfg, env_id=env_id)
        if observation == "full":
            indices = MATCHED_INDICES_FULL
        elif observation == "reference":
            indices = MATCHED_INDICES_REFERENCE
        else:
            raise ValueError(f"observation must be 'full' or 'reference', got {observation!r}")
        cfg = _matched_cfg(budgets, match_to, REFERENCE_REPEATS, activation, indices, hybrid_reference,
                           in_dim=len(indices) * REFERENCE_REPEATS)
        cfg["_observation"] = observation
        cfg.update(overrides)
        return "classic", cfg

    m = _MATCHED_REPLICATED.match(arm)
    if m:
        reps = int(m.group(1))
        budgets = pqc_parameter_budget(hybrid_reuse_config(reps), env_id=env_id)
        cfg = _matched_cfg(budgets, match_to, reps, activation, MATCHED_INDICES_FULL, f"hybrid_reuse{reps}",
                           in_dim=len(MATCHED_INDICES_FULL) * reps)
        cfg.update(overrides)
        return "classic", cfg

    m = _MATCHED_REUSE.match(arm)
    if m:
        reps = int(m.group(1))
        budgets = pqc_parameter_budget(hybrid_reuse_config(reps), env_id=env_id)
        cfg = _matched_cfg(budgets, match_to, REFERENCE_REPEATS, activation, MATCHED_INDICES_FULL, f"hybrid_reuse{reps}",
                           in_dim=len(MATCHED_INDICES_FULL) * REFERENCE_REPEATS)
        cfg.update(overrides)
        return "classic", cfg

    if arm == "frozenlake_matched_control":
        ref = hybrid_reference if hybrid_reference.startswith("frozenlake_") else FROZENLAKE_MATCHED_REFERENCE
        ref_type, ref_cfg = ARMS[ref]
        if ref_type != "hybrid":
            raise ValueError("hybrid_reference must name a hybrid arm")
        budgets = pqc_parameter_budget(resolve_transform(dict(ref_cfg)), env_id=FROZEN_SCALAR_ID)
        budget = budgets["total"] if match_to == "total" else budgets["quantum"]
        width, spent = match_hidden_width(budget, in_dim=1, out_dim=4)
        # ``match_hidden_width`` returns the closest width, which can fall under the budget of the hybrid: round up.
        while spent < budget and width < 4096:
            width += 1
            spent = 1 * width + width + width * 4 + 4
        cfg = {
            "net_arch": [width],
            "activation": activation,
            "_matched_to": ref,
            "_match_target": match_to,
            "_budget": budget,
            "_hybrid_total": budgets["total"],
            "_hybrid_quantum": budgets["quantum"],
            "_spent_params": spent,
        }
        cfg.update(overrides)
        return "classic", cfg

    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}. Known: {sorted(ARMS)}; derived: matched_control[_reuse<R>|_replicated<R>], "
                       "frozenlake_matched_control")
    agent_type, cfg = ARMS[arm]
    cfg = dict(cfg)
    cfg.update(overrides)
    return agent_type, resolve_transform(cfg)
