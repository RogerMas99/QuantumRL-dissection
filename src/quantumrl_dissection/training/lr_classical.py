"""Output Reuse hidden inside classical controls (docs/design/EXPERIMENT-13-OR-VS-LR.md, addendum).

E  The additive Fourier ceiling of CartPole (``FourierAdditiveCeiling``, 8 wires, L = 5) feeds the 4 observables to the
   8 wires cyclically, so wire i and wire i + 4 get the same variable and their features {cos kx, sin kx} are IDENTICAL
   columns. ``Linear(80 -> 2)`` (162 parameters) is therefore Output Reuse with R = 2 on a ``Linear(40 -> 2)`` (82).
F  The capacity-matched control of RQ2 (135 parameters) replicates the 4 observables x 4 before ``Linear(16 -> 7)``:
   Output Reuse with R = 4 on a ``Linear(4 -> 7)`` (51 parameters).
G  FrozenLake: a small ReLU MLP on the 4 bits {0, 1} of the state (the input of the binary circuit, with no angle map).

Config keys (read from the arm's ``agent_config``):

  lr_classical          True  -> use this module's post_build (``make_post_build`` of ``lr_equivalence`` delegates here)
  lr_classical_R        int   -> the model as built replicates its features R times (2 for the ceiling, 4 for the control)
  lr_collapse_R   int   -> replace the replicated head by the unreplicated one with V0 = sum of the R blocks
  lr_head_mult    float -> learning rate of the weight of the first linear layer x this factor (bias: base lr)
  lr_float64      True  -> float64 arithmetic (trace tests)
  lr_trace_every  int   -> keep effective parameters / actions every that many steps (trace tests)

The "head" is ``agent.head`` (ceiling) or the first top-level ``nn.Linear`` of ``agent.network`` (classical MLPs).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from quantumrl_dissection.training import lr_equivalence as L1

N_UNIQUE_WIRES = 4            # CartPole has 4 observables
BITS = 4                      # FrozenLake 4x4: ceil(log2(16))


# ---------------------------------------------------------------------------------------------------------------
# G: MLP on the bits of the state
# ---------------------------------------------------------------------------------------------------------------
class StateBits(nn.Module):
    """[B, 1] (state index as float) -> [B, 4] bits in {0, 1}, least significant first (the bit order of
    ``FrozenBasisToAngleTransformer``, without its factor pi)."""

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
    """``StateBits -> Linear(4, w) -> ReLU -> Linear(w, n_actions)``. Q-network only (DQN)."""

    def __init__(self, obs_dim: int, n_actions: int, config: Dict[str, Any], is_qnet: bool = True):
        super().__init__()
        if not is_qnet:
            raise ValueError("bits_mlp is DQN-only")
        w = int(config.get("width", 6))
        self.is_qnet = True
        self.network = nn.Sequential(StateBits(), nn.Linear(BITS, w), nn.ReLU(), nn.Linear(w, n_actions))

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------------------------------------------
# Locating the head in the three kinds of model
# ---------------------------------------------------------------------------------------------------------------
def head_linear(agent: nn.Module) -> nn.Linear:
    if isinstance(getattr(agent, "head", None), nn.Linear):          # FourierAdditiveCeiling
        return agent.head
    return agent.network[L1.head_index(agent.network)]


def effective_v(agent: nn.Module, R: int) -> torch.Tensor:
    """V_eff = sum of the R column blocks of the head weight (the weight itself when R = 1)."""
    w = head_linear(agent).weight
    out, inn = w.shape
    return w.reshape(out, R, inn // R).sum(dim=1) if R > 1 else w


def collapse_(agent: nn.Module, R: int) -> None:
    """In place: the model replicates its features R times; make it the model without replication with
    V0 = sum of the R blocks and the same bias. The torch RNG stream is left where it was."""
    old = head_linear(agent)
    out, inn = old.weight.shape
    f = inn // R
    with torch.random.fork_rng(devices=[]):
        new = nn.Linear(f, out, bias=old.bias is not None).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight.reshape(out, R, f).sum(dim=1))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    if isinstance(getattr(agent, "head", None), nn.Linear):          # ceiling: 8 wires -> 4 unique wires
        if agent.n_qubits != R * N_UNIQUE_WIRES:
            raise RuntimeError("collapse_ expects n_qubits == R * 4 for the Fourier ceiling")
        agent.n_qubits = N_UNIQUE_WIRES
        agent.head = new
        return
    net = agent.network
    i = L1.head_index(net)
    agent.network = nn.Sequential(*[new if j == i else m for j, m in enumerate(net)])
    ir = getattr(agent, "input_reuse", None)                         # ClassicalAgent: x -> x[:, idx].repeat(1, R)
    if ir is not None:
        if getattr(ir, "n_repeats", None) != R:
            raise RuntimeError("input_reuse.n_repeats does not match lr_classical_R")
        ir.n_repeats = 1


def build_adam(agent: nn.Module, base_lr: float, head_mult: float) -> torch.optim.Adam:
    hw = head_linear(agent).weight
    rest = [p for p in agent.parameters() if p is not hw]
    return torch.optim.Adam([{"params": [hw], "lr": base_lr * head_mult}, {"params": rest, "lr": base_lr}], lr=base_lr)


# ---------------------------------------------------------------------------------------------------------------
class EfgHook(L1.LearningRateHook):
    """The diagnostics of ``LearningRateHook`` for models whose head is not the first Linear of ``network`` or whose features
    are replicated by a module outside ``network``."""

    def __init__(self, dqn, R: int, **kw):
        self.R = R
        super().__init__(dqn, **kw)

    def _flat_effective(self):
        q = self.dqn.q_network
        hw, parts = head_linear(q), []
        for p in q.parameters():
            parts.append((effective_v(q, self.R_now()) if p is hw.weight else p).detach().reshape(-1))
        return torch.cat(parts).double().cpu().numpy()

    def R_now(self) -> int:
        """R of the model as it is now (1 after the collapse)."""
        return self.R

    def on_step(self, n: int) -> None:
        if self.trace_every and n % self.trace_every == 0:
            self.trace_steps.append(n)
            self.trace_params.append(self._flat_effective())
        if n % self.every:
            return
        rb, q = self.dqn.rb, self.dqn.q_network
        size = rb.max_size if rb.full else rb.pos
        k = min(self.n_states, size)
        import numpy as np
        idx = (np.arange(rb.pos - k, rb.pos) % rb.max_size) if k else np.arange(0)
        with torch.no_grad():
            vfro = float(torch.linalg.norm(effective_v(q, self.R).double()))
            mq = float(q(torch.as_tensor(rb.obs_buf[idx], dtype=torch.get_default_dtype())).abs().double().mean()) \
                if k else float("nan")
        self.rows.append((n, vfro, mq, self.last_td))


def make_post_build(cfg: Dict[str, Any]):
    R_built = int(cfg.get("lr_classical_R", 1))
    collapse = int(cfg.get("lr_collapse_R", 0) or 0)
    head_mult = float(cfg.get("lr_head_mult", 1.0))
    f64 = bool(cfg.get("lr_float64", False))
    trace_every = int(cfg.get("lr_trace_every", 0) or 0)

    def post_build(dqn) -> EfgHook:
        dqn.envs.single_action_space.seed(int(dqn.seed))          # same reason as lr_equivalence.make_post_build
        q = dqn.q_network
        if f64:
            L1.to_float64_(q)
        R_now = R_built
        if collapse:
            if collapse != R_built:
                raise RuntimeError(f"lr_collapse_R={collapse} but lr_classical_R={R_built}")
            collapse_(q, collapse)
            R_now = 1
        dqn.target_network = copy.deepcopy(q)
        if head_mult != 1.0:
            dqn.optimizer = build_adam(q, dqn.learning_rate, head_mult)
        elif f64 or collapse:
            dqn.optimizer = torch.optim.Adam(q.parameters(), lr=dqn.learning_rate)
        return EfgHook(dqn, R_now, trace_every=trace_every)

    return post_build


# ---------------------------------------------------------------------------------------------------------------
# Arm registry (called from arms.py)
#
# ``fourier_ceiling`` (E) and ``matched_control`` (F) are the arms of the main campaign: in this experiment they run with
# the stage overrides lr_classical = true and lr_classical_R = 2 and 4 respectively (the model as built replicates its
# features that many times), which only switch on the diagnostics; the model is the same.
# ---------------------------------------------------------------------------------------------------------------
E_ARMS = ("fourier_ceiling", "fourier_ceiling_distinct", "fourier_ceiling_distinct_lr_x2",
          "fourier_ceiling_distinct_lr_x2_collapsed")
F_ARMS = ("matched_control", "network_no_replication_w7_lr_x4", "network_no_replication_w7_lr_x4_collapsed",
          "network_no_replication_w18", "network_no_replication_w18_lr_x4")
G_ARMS = ("frozenlake_bits_network_w6", "frozenlake_bits_network_w12")
BLOCK_ARMS = {"E": E_ARMS, "F": F_ARMS, "G": G_ARMS}
ENV_OF_BLOCK = {"E": "CartPole-v1", "F": "CartPole-v1", "G": None}        # G: FROZEN_SCALAR_ID, resolved by the runner


def arm_definitions(hybrid: Dict[str, Any], reuse_indices) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    base = {"n_qubits": hybrid["n_qubits"], "n_layers_q": hybrid["n_layers_q"], "circ_type": "skolik",
            "lr_classical": True}
    uniq = dict(base, n_qubits=N_UNIQUE_WIRES)
    rep4 = {"reuse_indices": list(reuse_indices), "n_repeats": 4, "net_arch": [7], "activation": nn.ReLU,
            "lr_classical": True, "lr_classical_R": 4}
    norep = {"reuse_indices": list(reuse_indices), "n_repeats": 1, "activation": nn.ReLU, "lr_classical": True}
    bits = {"lr_classical": True}
    return {
        "fourier_ceiling_distinct": ("fourier_additive", dict(uniq)),
        "fourier_ceiling_distinct_lr_x2": ("fourier_additive", dict(uniq, lr_head_mult=2.0)),
        "fourier_ceiling_distinct_lr_x2_collapsed": ("fourier_additive", dict(base, lr_classical_R=2, lr_collapse_R=2,
                                                                              lr_head_mult=2.0)),
        "network_no_replication_w7_lr_x4": ("classic", dict(norep, net_arch=[7], lr_head_mult=4.0)),
        "network_no_replication_w7_lr_x4_collapsed": ("classic", dict(rep4, lr_collapse_R=4, lr_head_mult=4.0)),
        "network_no_replication_w18": ("classic", dict(norep, net_arch=[18])),
        "network_no_replication_w18_lr_x4": ("classic", dict(norep, net_arch=[18], lr_head_mult=4.0)),
        "frozenlake_bits_network_w6": ("bits_mlp", dict(bits, width=6)),
        "frozenlake_bits_network_w12": ("bits_mlp", dict(bits, width=12)),
    }
