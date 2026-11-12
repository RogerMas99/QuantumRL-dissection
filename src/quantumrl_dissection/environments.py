"""
Observation-space adapters and environment registration.

NAMED `obs_adapters`, NOT `envs`, on purpose: the vendored library already has a
`simplyqrl/envs.py`, and that one is a vector-env factory (`make_vec_env`) with
nothing to do with observation spaces. Two files called `envs.py` in one tree is
a trap for a reader, and this module does something narrower than "environments"
anyway - it changes how an observation is presented, never what the environment
does.

Why this module exists
----------------------
Upstream SimplyQRL is written throughout for `Box` observation spaces. Under
DQN a `Discrete` observation space (FrozenLake, Taxi, CliffWalking) breaks the
pipeline in three independent places, because `Discrete(n).shape == ()`:

  1. Action selection. `self.obs` from the single-env `SyncVectorEnv` has shape
     `(1,)`, so `nn.Linear` reads it as one sample with one feature and returns
     a 1-D `q_values`; `torch.argmax(q_values, dim=1)` then raises IndexError on
     the first step.
  2. Replay buffer. `np.zeros((buffer_size, *observation_space.shape),
     dtype=observation_space.dtype)` allocates `(buffer_size,)` `int64`. A
     sampled batch is `[B]`, one-dimensional and integer, where every downstream
     consumer expects `[B, 1]` float.
  3. Silently. The built-in `Frozen*` transformers branch on `data.dim() == 1`
     to handle a single observation; given a `[B]` batch they take that branch
     and return one sample's angles for the whole batch. No exception. And
     `FrozenNormalizationTransformer` clones an int64 tensor and writes a float
     into it, truncating the encoding angle.

PPO escapes all three: it reshapes to `(num_envs, -1)` explicitly and stores
observations in a float tensor - with the CartPole-only version surviving as a
comment immediately above. The same thing happened to the off-policy path, in a second
subsystem: the on-policy path was hardened when the library moved beyond
CartPole, the off-policy path was not re-validated.

The adjustment, and why it is shaped this way
--------------------------------------
We do not patch upstream internals. We adapt the environment so it presents its
observation in the form the rest of the stack already assumes: `Box(shape=(1,),
dtype=float32)`, the state index unchanged in value. This reproduces PPO's
effective representation exactly, so a future cross-algorithm comparison sees
the same thing on both sides.

Registering gym ids rather than passing env objects is what keeps the blast
radius at zero: `SafeDQN` does `gym.make(self.env_id)` and `run_arm`/`run_grid`
take an `env_id` string, so **no runner code changes**.

Placement is `core/`, not `dqn/`: only DQN currently needs it, but it is an
environment concern and PPO can use it unharmed. Per the repo rule, a thing that
would be copied from `dqn/` to `ppo/` belonged in `core/` to begin with.

Evidence and scope: docs/adjustments.md.
Regression tests: tests/test_frozenlake_envs.py.
"""

from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium import spaces

__all__ = [
    "DiscreteToBoxObs",
    "OneHotObs",
    "register_environments",
    "FROZEN_SCALAR_ID",
    "FROZEN_ONEHOT_ID",
]

FROZEN_SCALAR_ID = "FrozenLake4x4Scalar-v0"
FROZEN_ONEHOT_ID = "FrozenLake4x4OneHot-v0"

# The FrozenLake-v1 4x4 default TimeLimit, pinned explicitly rather than
# inherited. Truncation is where the phantom transition of the autoreset is generated, so
# this boundary must be ours and not a gymnasium default that could move.
FROZEN_MAX_EPISODE_STEPS = 100

_FROZEN_KWARGS = {"map_name": "4x4", "is_slippery": False}


class DiscreteToBoxObs(gym.ObservationWrapper):
    """Present `Discrete(n)` as `Box(shape=(1,), float32)`.

    The state index is passed through unchanged in value; only the container
    changes. Normalisation into angles stays where it belongs - in the agent's
    `transform_fn` - so the chapter's `FrozenNormalizationTransformer` and
    `FrozenBasisToAngleTransformer` keep doing exactly what they document.
    """

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
    """Present `Discrete(n)` as a one-hot `Box(shape=(n,), float32)`.

    For the classical reference arm only. It gives the agent a representation
    with no spurious metric between state indices, which is the strongest
    classical baseline available on a grid-world - and therefore the right arm to
    answer "is this environment learnable by this stack at all?" before spending
    PQC compute. an earlier design lost two experiments to measuring architecture effects on
    agents that were not learning.
    """

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
    """Register the adapted FrozenLake ids. Idempotent; safe to call repeatedly.

    Called at `core.configs` import time so the ids exist as soon as the package
    does, which is what lets `SafeDQN(env_id=...)` and `run_grid(env_id=...)`
    work with no changes.
    """
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


# ---------------------------------------------------------------------------------------------------------------------
# Selecting components of the observation in the environment
# ---------------------------------------------------------------------------------------------------------------------
REFERENCE_COMPONENTS = (1, 2, 3)        # the reference study's classical control discards the cart position


class SelectObs(gym.ObservationWrapper):
    """Keep only ``indices`` of a 1-D Box observation.

    The vendored library applies ``reuse_indices`` only when the observation is replicated, so a model that reads the
    components [1, 2, 3] a single time receives them from the environment instead.
    """

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
