"""The classical twin of the hybrid Q-network: same head, same Output Reuse, classical feature extractor.

The hybrid agent (upstream ``HybridAgent`` with ``is_qnet=True``) is

    obs (4) -> PQC (80 weights, 8 <Z_i> in [-1, 1]) -> OutputReuse(R) -> Linear(8R -> 4) -> Identity -> Linear(4 -> 2)

Output Reuse acts only at the border between the circuit and the linear head, so a faithful classical control
keeps EVERYTHING after the circuit and replaces only the circuit by a small classical extractor with the same
number of outputs and (for the bounded variant) the same range:

    obs (4) -> Linear(4 -> 6) -> ReLU -> Linear(6 -> 8) [-> Tanh] -> OutputReuse(R) -> the same head

How the twin is built. ``ClassicalTwinAgent`` IS a ``HybridAgent``: it calls the upstream constructor with the
very config of ``hybrid_reuse{R}`` (so OutputReuse, the head layers, their default ``nn.Linear`` initialisation, the
activation and the ``forward`` are the upstream objects and the upstream code, not a copy), and then swaps the
first module of ``self.network`` (the PennyLane layer) for the classical extractor. Consequences:

* the head of the twin is built by the same code, in the same order, from the same RNG state as the head of the
  hybrid, so under one seed the two heads start with identical weights (tested);
* the optimiser and the DQN loop see an ordinary ``HybridAgent``; nothing in the vendored library is modified;
* the PennyLane layer is built and then discarded (a few milliseconds); it is the price of not copying the head.

Only DQN Q-networks are supported (``is_qnet=True``).

Config keys, on top of a ``hybrid_or_config(R)`` dictionary: ``twin_hidden`` (default 6) and ``twin_bounded``
(default True: final Tanh, outputs in [-1, 1] like the circuit's <Z_i>; False: no Tanh, outputs unbounded).
"""
from __future__ import annotations

from torch import nn

from simplyqrl.agents import HybridAgent

TWIN_HIDDEN = 6          # extractor: 4 -> 6 -> 8 = 30 + 56 = 86 parameters


class ClassicalTwinAgent(HybridAgent):
    """``HybridAgent`` whose quantum layer is replaced by a classical extractor (see module docstring)."""

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
                               "self.network; review core/classical_twin.py before using it.")
        layers = [nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.n_qubits)]
        if bounded:
            layers.append(nn.Tanh())
        self.twin_bounded = bounded
        net[0] = nn.Sequential(*layers)      # same position, same number of outputs as the circuit
        del self.qnet_q_layer                # drop the second registration of the discarded PennyLane layer

    @property
    def extractor(self) -> nn.Module:
        return self.network[0]

    @property
    def head(self) -> nn.Sequential:
        """Everything after the extractor: [OutputReuse(R) if R > 1] -> Linear -> activation -> Linear."""
        return self.network[1:]
