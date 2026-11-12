"""
quantumrl_dissection: component-based evaluation of parametrized quantum circuits in reinforcement learning, from PPO
to DQN, across configurations and environments.

Built on top of SimplyQRL (Lazaro, Vazquez & Garcia Bringas, 2025), vendored unmodified in ``third_party/simplyqrl``.
The adjustments that this work applies to the library are applied at import time by ``training.adjustments`` and are
documented in ``docs/adjustments.md``.

Layout
------
    circuits/     circuit templates and their configurations
    agents/       parameter accounting, the classical twin, the additive Fourier ceilings
    simulators/   SU(2) emulator and batched state-vector simulator
    training/     DQN wrapper (terminal adjustment, probe, greedy evaluation), runs, PPO, learning-rate equivalence
    analysis/     loaders, statistics, tables, figures
    arms.py       the registry: arm name -> configuration
"""

__version__ = "1.0.0"

from quantumrl_dissection.training.adjustments import apply_upstream_patches, upstream_report

# The adjustments patch the library itself, so they belong at package import.
apply_upstream_patches()

from quantumrl_dissection.agents.capacity import capacity_ladder, count_trainable, match_hidden_width  # noqa: E402
from quantumrl_dissection.arms import ARMS, build_arm_config  # noqa: E402
from quantumrl_dissection.environments import (  # noqa: E402
    FROZEN_ONEHOT_ID, FROZEN_SCALAR_ID, register_environments,
)

__all__ = [
    "ARMS", "build_arm_config",
    "register_environments", "FROZEN_SCALAR_ID", "FROZEN_ONEHOT_ID",
    "capacity_ladder", "count_trainable", "match_hidden_width",
    "upstream_report",
]
