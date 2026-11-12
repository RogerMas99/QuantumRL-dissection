"""
Source-level corrections applied to the installed SimplyQRL.

Design note
-----------
These are applied at import time by rebinding attributes on the loaded modules,
not by editing files on disk. That keeps the upstream checkout pristine and
citable, and makes the diff between "their code" and "our code" exactly this
file.

Every patch is GUARDED: it first asserts that upstream still looks the way the
patch expects. If upstream changes, we raise loudly rather than silently
applying a no-op - which is the very failure mode we are correcting.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import torch.nn as nn

_APPLIED: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# agent_type="classic" is not resolvable by build_agent
#
# Upstream:  simplyqrl/agents.py :: build_agent
#            if type == "mlp": ...  elif type == "hybrid": ...
#            No else branch, no raise -> returns None for any other name.
# Symptom:   src/experiments/post-pqc-inference.py:53 passes agent_type="classic"
#            (the paper's classical control arm). build_agent returns None and
#            the next line, self.agent.to(device), raises AttributeError.
#            That published experiment script does not run as distributed.
# Fix:       alias "classic" -> "mlp"; raise on unknown names.
# See:       docs/adjustments.md
# ---------------------------------------------------------------------------
_AGENT_TYPE_ALIASES = {
    "classic": "mlp", "classical": "mlp", "mlp": "mlp",
    "hybrid": "hybrid", "quantum": "hybrid",
}


def _patch_agent_type_aliases() -> None:
    from simplyqrl import agents as up

    original = up.build_agent
    if getattr(original, "_quantumrl_dissection_patched", False):
        return

    src = inspect.getsource(original)
    if 'if type == "mlp"' not in src or 'elif type == "hybrid"' not in src:
        raise RuntimeError(
            "guard of the agent-type aliases failed: upstream build_agent no longer dispatches on "
            '"mlp"/"hybrid" as expected. Review docs/adjustments.md '
            "against the pinned SimplyQRL revision before proceeding."
        )

    def build_agent(type, obs_shape, n_actions, config, is_qnet=False):  # noqa: A002
        key = str(type).lower()
        if key not in _AGENT_TYPE_ALIASES:
            raise ValueError(
                "unknown agent_type %r. Known: %s"
                % (type, sorted(set(_AGENT_TYPE_ALIASES)))
            )
        return original(_AGENT_TYPE_ALIASES[key], obs_shape, n_actions,
                        config, is_qnet=is_qnet)

    build_agent._quantumrl_dissection_patched = True
    up.build_agent = build_agent

    # dqn.py and ppo.py do `from .agents import build_agent`, so rebinding the
    # attribute on `agents` alone would not reach them.
    for mod_name in ("simplyqrl.dqn", "simplyqrl.ppo"):
        try:
            mod = __import__(mod_name, fromlist=["build_agent"])
        except Exception:  # pragma: no cover - optional module
            continue
        if hasattr(mod, "build_agent"):
            mod.build_agent = build_agent

    _APPLIED["agent_type_aliases"] = 'agent_type aliases: "classic"/"classical" -> "mlp"'


# ---------------------------------------------------------------------------
# OutputScale is a no-op in the is_qnet branch
#
# Upstream:  simplyqrl/agents.py :: HybridAgent.__init__ (is_qnet branch)
#
#                qnet_layers.append(nn.Linear(prev_dim, act_dim))
#                self.network = nn.Sequential(*qnet_layers)      # <- consumed
#
#                if config.get("use_output_scaling", False):
#                    init_val = config.get("output_scale_init", 2.0)
#                    qnet_layers.append(OutputScale(act_dim, init_val))  # too late
#
#            The scaling layer is appended to a list nn.Sequential already
#            consumed, so use_output_scaling never reaches the model.
# Scope:     DQN only. PPO is unaffected (softmax is scale-invariant), so the
#            paper's published results do not depend on this.
# Why it     A PQC readout is bounded in [-1, 1] while CartPole Q-values reach
# matters:   ~1/(1-gamma) = 100. Without scaling, the linear head must learn very
#            large weights - exactly what output scaling exists to avoid
#            (Skolik et al. 2022).
# Fix:       rebuild self.network with OutputScale appended, post-construction.
# See:       docs/adjustments.md
# ---------------------------------------------------------------------------
def _patch_output_scale() -> None:
    from simplyqrl import agents as up

    if getattr(up.HybridAgent, "_quantumrl_dissection_output_scale_patched", False):
        return

    src = inspect.getsource(up.HybridAgent.__init__)
    if "use_output_scaling" not in src or "OutputScale" not in src:
        raise RuntimeError(
            "guard of the output scale failed: upstream HybridAgent no longer mentions "
            "use_output_scaling/OutputScale. Review docs/adjustments.md."
        )

    original_init = up.HybridAgent.__init__

    def __init__(self, obs_dim, act_dim, config, is_qnet=False):
        original_init(self, obs_dim, act_dim, config, is_qnet=is_qnet)
        if not is_qnet:
            return
        cfg = config or {}
        if not cfg.get("use_output_scaling", False):
            return
        net = getattr(self, "network", None)
        if not isinstance(net, nn.Sequential):
            raise RuntimeError("output scale: expected self.network to be nn.Sequential")
        if any(isinstance(m, up.OutputScale) for m in net):
            return  # upstream corrected it; nothing to do
        init_val = cfg.get("output_scale_init", 2.0)
        self.network = nn.Sequential(*list(net), up.OutputScale(act_dim, init_val))

    up.HybridAgent.__init__ = __init__
    up.HybridAgent._quantumrl_dissection_output_scale_patched = True
    _APPLIED["output_scale"] = "OutputScale now actually appended to HybridAgent.network"


# ---------------------------------------------------------------------------
# agent_type dispatch extension: su2, fourier_additive,
# linear_on_bits
#
# Not a correction - an ADDITION. Wraps whatever `build_agent` already is
# (after the agent-type aliases above) with three agent_type values this project
# defines itself, never present upstream: they construct `SU2HybridAgent` /
# `FourierAdditiveCeiling` / `linear_on_bits_ceiling` directly instead of
# falling through to upstream's mlp/hybrid dispatch. Anything not in this set
# is passed through UNCHANGED to whatever build_agent was before this patch,
# so `_patch_agent_type_aliases`'s own code above is never touched.
#
# These three are inference/verification stand-ins (core/su2_emulator.py,
# core/fourier_ceiling.py), not new experimental hypotheses of their own -
# see those modules' docstrings for what each does and does not claim.
# See:       docs/adjustments.md, docs/adjustments.md
# ---------------------------------------------------------------------------
_NEW_AGENT_TYPES = {"su2", "sv", "fourier_additive", "linear_on_bits", "classical_twin", "bits_mlp"}


def _patch_new_agent_types() -> None:
    import numpy as np
    from simplyqrl import agents as up

    previous = up.build_agent
    if getattr(previous, "_quantumrl_dissection_new_types_patched", False):
        return

    def build_agent(type, obs_shape, n_actions, config, is_qnet=False):  # noqa: A002
        key = str(type).lower()
        if key in _NEW_AGENT_TYPES:
            if not is_qnet:
                raise ValueError(
                    f"agent_type={key!r} only supports is_qnet=True (DQN); it "
                    "has no actor-critic construction."
                )
            cfg = config or {}
            obs_dim = int(np.prod(obs_shape))
            if key == "su2":
                from quantumrl_dissection.simulators.emulator import SU2HybridAgent
                return SU2HybridAgent(obs_dim, n_actions, cfg, is_qnet=True)
            if key == "sv":
                from quantumrl_dissection.simulators.statevector import SVHybridAgent
                return SVHybridAgent(obs_dim, n_actions, cfg, is_qnet=True)
            if key == "bits_mlp":                                  # MLP on the 4 bits of the state
                from quantumrl_dissection.training.lr_classical import BitsMLPAgent
                return BitsMLPAgent(obs_dim, n_actions, cfg, is_qnet=True)
            if key == "classical_twin":
                from quantumrl_dissection.agents.classical_twin import ClassicalTwinAgent
                return ClassicalTwinAgent(obs_dim, n_actions, cfg, is_qnet=True)
            if key == "fourier_additive":
                from quantumrl_dissection.agents.fourier_ceiling import FourierAdditiveCeiling
                return FourierAdditiveCeiling(
                    n_qubits=cfg.get("n_qubits", 4),
                    n_layers=cfg.get("n_layers_q", 1),
                    n_actions=n_actions,
                    emb_indices=cfg.get("emb_indices", None),
                    transform_fn=cfg.get("transform_fn", None),
                    circ_type=cfg.get("circ_type", "skolik"),
                    n_data=obs_dim,
                )
            if key == "linear_on_bits":
                from quantumrl_dissection.agents.fourier_ceiling import linear_on_bits_ceiling
                return linear_on_bits_ceiling(
                    cfg.get("n_qubits", 4), n_actions,
                    transform_fn=cfg.get("transform_fn", None),
                )
        return previous(type, obs_shape, n_actions, config, is_qnet=is_qnet)

    build_agent._quantumrl_dissection_new_types_patched = True
    up.build_agent = build_agent

    # Same rebinding as the agent-type aliases above - dqn.py and ppo.py imported build_agent
    # by name, so the attribute on `agents` alone would not reach them.
    for mod_name in ("simplyqrl.dqn", "simplyqrl.ppo"):
        try:
            mod = __import__(mod_name, fromlist=["build_agent"])
        except Exception:  # pragma: no cover - optional module
            continue
        if hasattr(mod, "build_agent"):
            mod.build_agent = build_agent

    _APPLIED["simulator_and_ceiling_agent_types"] = (
        "agent_type dispatch extended: su2, fourier_additive, linear_on_bits"
    )


def apply_upstream_patches() -> None:
    """Idempotent. Called on `import quantumrl_dissection`."""
    _patch_agent_type_aliases()
    _patch_output_scale()
    _patch_new_agent_types()


def upstream_report() -> Dict[str, Any]:
    """What was patched, and against which upstream revision."""
    import gymnasium
    import simplyqrl

    try:
        import pennylane
        pl = pennylane.__version__
    except Exception:  # pragma: no cover
        pl = "not installed"

    return {
        "patches_applied": dict(_APPLIED),
        "simplyqrl": getattr(simplyqrl, "__file__", "?"),
        "gymnasium": gymnasium.__version__,
        "pennylane": pl,
    }
