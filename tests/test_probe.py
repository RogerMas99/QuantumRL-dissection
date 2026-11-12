"""the terminal adjustment / the probe: phantom detection and skipping.

Runs against a fake DQN with the same call shape as upstream's training loop,
so it needs gymnasium but neither torch nor pennylane. Fast enough for CI.
"""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import SyncVectorEnv

from conftest import load_by_path

# Loaded by path on purpose: importing quantumrl_dissection.dqn.safe would execute the package
# __init__, which pulls torch and simplyqrl. The probe needs neither, and keeping
# this test light means it runs in CI without a quantum stack.
AutoresetProbe = load_by_path("_safe", "src/quantumrl_dissection/training/dqn.py").AutoresetProbe


class _FakeBuffer:
    def __init__(self):
        self.rows = []

    def add(self, obs, next_obs, action, reward, done, infos):
        self.rows.append(float(np.asarray(reward).reshape(-1)[0]))


class _FakeDQN:
    def __init__(self):
        self.envs = SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        self.rb = _FakeBuffer()


def _roll(skip_phantom, n=3000, seed=0):
    dqn = _FakeDQN()
    probe = AutoresetProbe(dqn, skip_phantom=skip_phantom)
    obs, _ = dqn.envs.reset(seed=seed)
    for _ in range(n):
        actions = dqn.envs.action_space.sample()
        next_obs, reward, term, trunc, infos = dqn.envs.step(actions)
        dqn.rb.add(obs, next_obs.copy(), actions, reward, term, infos)
        obs = next_obs
    probe.restore()
    dqn.envs.close()
    return probe, dqn


def test_autoreset_mode_is_next_step():
    """If this fails, the terminal adjustment is a no-op on this gymnasium and results change."""
    env = SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    mode = str(env.metadata.get("autoreset_mode", ""))
    env.close()
    assert "NEXT_STEP" in mode, f"unexpected autoreset mode: {mode}"


def test_phantoms_are_detected():
    probe, _ = _roll(skip_phantom=False)
    assert probe.n_steps == 3000
    assert probe.n_phantom > 0
    assert 0.0 < probe.frac_poison < 0.5


def test_reward_zero_is_an_independent_phantom_detector():
    """In CartPole reward is always 1.0 while stepping, so reward==0 marks the
    phantom exactly. Two independent detectors must agree."""
    probe, dqn = _roll(skip_phantom=False)
    n_reward_zero = sum(1 for r in dqn.rb.rows if r == 0.0)
    assert n_reward_zero == probe.n_phantom


def test_patch_removes_every_phantom_from_the_buffer():
    probe, dqn = _roll(skip_phantom=True)
    assert probe.n_phantom > 0, "nothing to skip - test is vacuous"
    assert sum(1 for r in dqn.rb.rows if r == 0.0) == 0
    assert len(dqn.rb.rows) == probe.n_added


def test_summary_reports_stored_phantoms_consistently():
    on, _ = _roll(skip_phantom=True)
    off, _ = _roll(skip_phantom=False)
    assert on.summary()["phantoms_stored"] == 0
    assert off.summary()["phantoms_stored"] == off.n_phantom
