"""the discrete-observation adapter: Discrete observation spaces under DQN.

Two jobs, deliberately separated.

`test_upstream_*` PIN THE BUG: they assert that raw FrozenLake still produces the
malformed shape and dtype. If one of these ever fails, upstream has repaired the
problem and `core/obs_adapters.py` may be removable - which is the signal we want, and
the reason the bug is asserted rather than described in a comment.

`test_wrapped_*` assert the adapter produces what the stack assumes.

Loaded by path so the file runs without torch, pennylane or simplyqrl, like the
other light tests. `core/obs_adapters.py` needs only gymnasium and numpy.
"""
import numpy as np
import pytest

import gymnasium as gym

from conftest import load_by_path

envs = load_by_path("_envs", "src/quantumrl_dissection/environments.py")

FROZEN = dict(map_name="4x4", is_slippery=False)


def upstream_allocation(space, buffer_size=1000):
    """Verbatim reproduction of simplyqrl/buffers.py::ReplayBuffer.__init__."""
    return np.zeros((buffer_size, *space.shape), dtype=space.dtype)


# ---------------------------------------------------------------------------
# The bug, pinned
# ---------------------------------------------------------------------------

def test_upstream_discrete_space_has_no_feature_axis():
    space = gym.make("FrozenLake-v1", **FROZEN).observation_space
    assert space.shape == ()
    assert space.dtype == np.int64


def test_upstream_buffer_allocation_is_one_dimensional():
    """A sampled batch would be `[B]`, not `[B, 1]`.

    The built-in Frozen transformers take their `data.dim() == 1` branch on such
    a tensor and return one sample's angles for the whole batch - silently wrong
    rather than loudly broken, which is why this needs a test and not a comment.
    """
    space = gym.make("FrozenLake-v1", **FROZEN).observation_space
    buf = upstream_allocation(space)
    assert buf.shape == (1000,)
    assert buf.dtype == np.int64
    assert buf[np.arange(128)].ndim == 1


def test_upstream_cartpole_is_unaffected():
    """Contrast arm: CartPole allocates correctly, which is why the bug survived."""
    buf = upstream_allocation(gym.make("CartPole-v1").observation_space)
    assert buf.shape == (1000, 4)
    assert buf[np.arange(128)].ndim == 2


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

def test_scalar_wrapper_shape_and_dtype():
    env = envs.DiscreteToBoxObs(gym.make("FrozenLake-v1", **FROZEN))
    assert env.observation_space.shape == (1,)
    assert env.observation_space.dtype == np.float32

    obs, _ = env.reset(seed=0)
    assert obs.shape == (1,) and obs.dtype == np.float32

    obs = env.step(env.action_space.sample())[0]
    assert obs.shape == (1,) and obs.dtype == np.float32


def test_scalar_wrapper_preserves_the_state_value():
    """The container changes, never the value: FrozenLake starts at cell 0."""
    env = envs.DiscreteToBoxObs(gym.make("FrozenLake-v1", **FROZEN))
    obs, _ = env.reset(seed=0)
    assert float(obs[0]) == 0.0


def test_onehot_wrapper():
    env = envs.OneHotObs(gym.make("FrozenLake-v1", **FROZEN))
    assert env.observation_space.shape == (16,)
    obs, _ = env.reset(seed=0)
    assert obs.sum() == 1.0 and obs.argmax() == 0


def test_wrappers_reject_a_box_space():
    with pytest.raises(TypeError):
        envs.DiscreteToBoxObs(gym.make("CartPole-v1"))


def test_buffer_allocation_is_correct_after_wrapping():
    env = envs.DiscreteToBoxObs(gym.make("FrozenLake-v1", **FROZEN))
    buf = upstream_allocation(env.observation_space)
    assert buf.shape == (1000, 1)
    assert buf.dtype == np.float32
    assert buf[np.arange(128)].ndim == 2


# ---------------------------------------------------------------------------
# Registration - this is what lets SafeDQN and run_grid stay unchanged
# ---------------------------------------------------------------------------

def test_registration_is_idempotent():
    first = envs.register_environments()
    second = envs.register_environments()
    assert first == second == [envs.FROZEN_SCALAR_ID, envs.FROZEN_ONEHOT_ID]


def test_registered_ids_build_the_right_spaces():
    envs.register_environments()
    assert gym.make(envs.FROZEN_SCALAR_ID).observation_space.shape == (1,)
    assert gym.make(envs.FROZEN_ONEHOT_ID).observation_space.shape == (16,)
    assert gym.make(envs.FROZEN_SCALAR_ID).action_space.n == 4


def test_registered_env_pins_the_time_limit():
    """Truncation is where the the terminal adjustment phantom transition is born, so the
    boundary must be ours and not a gymnasium default that could move."""
    envs.register_environments()
    env = gym.make(envs.FROZEN_SCALAR_ID)
    assert env.spec.max_episode_steps == envs.FROZEN_MAX_EPISODE_STEPS == 100


def test_episodes_are_short_which_is_the_point():
    """the premise of the FrozenLake experiment: the phantom fraction is ~1/mean_episode_length, and on
    FrozenLake that stays large for the whole run instead of shrinking as the
    agent improves (which is why the terminal adjustment was unmeasurable on CartPole)."""
    envs.register_environments()
    env = gym.make(envs.FROZEN_SCALAR_ID)
    rng = np.random.default_rng(0)
    lengths = []
    for _ in range(200):
        env.reset(seed=int(rng.integers(1 << 30)))
        done, n = False, 0
        while not done:
            _, _, term, trunc, _ = env.step(int(rng.integers(4)))
            n += 1
            done = term or trunc
        lengths.append(n)
    mean_len = float(np.mean(lengths))
    assert mean_len < 25, f"expected short episodes, got mean {mean_len:.1f}"
    assert 1.0 / mean_len > 0.04, "phantom fraction should dwarf CartPole's <0.01"
