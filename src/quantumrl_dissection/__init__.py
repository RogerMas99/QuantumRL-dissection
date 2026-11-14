"""Component-based evaluation of parametrized quantum circuits in reinforcement learning (DQN and PPO, CartPole and FrozenLake)."""
from quantumrl_dissection.training import apply_upstream_patches, upstream_report

__all__ = ["apply_upstream_patches", "upstream_report"]
__version__ = "1.0.0"

# The adjustments patch the library itself, so they are applied when the package is imported.
apply_upstream_patches()
