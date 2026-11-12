"""
SafeDQN - the upstream DQN with the terminal adjustment applied and instrumented.

The terminal adjustment and the probe share one mechanism, deliberately: the same wrapper that can
*skip* the phantom transition also *counts* it. That means every run reports the
poisoning rate it was exposed to, including runs where the adjustment is off. A patch
you cannot measure is a patch you cannot defend.
"""

from __future__ import annotations

import contextlib
import csv
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

# torch and simplyqrl are imported lazily inside the functions that need them,
# so AutoresetProbe - the core instrument - can be unit-tested without the full
# quantum stack. See tests/test_probe.py.


# ---------------------------------------------------------------------------
# / Autoreset NEXT_STEP phantom transition
#
# Upstream:  simplyqrl/dqn.py, training loop:
#
#                real_next_obs = next_obs.copy()
#                # Save to replay buffer
#                self.rb.add(self.obs, real_next_obs, actions, rewards,
#                            terminations, infos)
#
#            The call is unconditional. Since gymnasium >= 1.0 the default
#            autoreset mode of SyncVectorEnv is AutoresetMode.NEXT_STEP: the
#            step() following a termination ignores the action, resets, and
#            returns reward=0, terminated=False with the reset observation.
#            That PHANTOM transition therefore enters the replay buffer.
#
# Effect:    it teaches Q(terminal) ~= gamma * V(s0) - a value leak through
#            death: dying leads, according to the buffer, to a fresh initial
#            state carrying all its future value. Consistent with a stable loss
#            (targets are finite and mutually consistent) and with a collapse
#            that deepens as the agent improves (the leak is worth gamma*V(s0),
#            which grows).
#
# Scope:     off-policy only. PPO shares the phantom transition but has no
#            replay buffer, so the effect is a bounded point bias in GAE at the
#            episode boundary rather than cumulative poisoning.
#
# Detection: the phantom is always the rb.add() immediately following a step
#            that reported terminated or truncated. Wrapping envs.step (rather
#            than reading the `done` argument rb.add receives, which carries
#            only `terminations`) also catches phantoms after TRUNCATION -
#            relevant as soon as the agent starts reaching the 500-step cap.
#
# See:       docs/adjustments.md
# ---------------------------------------------------------------------------
class AutoresetProbe:
    """Wraps envs.step and rb.add on a constructed DQN.

    Parameters
    ----------
    skip_phantom : bool
        True  -> the terminal adjustment active: phantom transitions never reach the buffer.
        False -> upstream behaviour, faithfully reproduced, still measured.
    """

    def __init__(self, dqn: Any, skip_phantom: bool = True):
        self.dqn = dqn
        self.skip_phantom = skip_phantom
        self.n_steps = 0
        self.n_added = 0
        self.n_phantom = 0
        self.trace: List[int] = []
        self._pending = False
        self._prev = False
        self._orig_step = dqn.envs.step
        self._orig_add = dqn.rb.add
        dqn.envs.step = self._step
        dqn.rb.add = self._add
        self.on_step = None  # optional callback(step_index) -> None

    def _step(self, actions):
        out = self._orig_step(actions)
        _, _, term, trunc, _ = out
        self._pending = bool(np.any(term)) or bool(np.any(trunc))
        return out

    def _add(self, obs, next_obs, action, reward, done, infos):
        is_phantom = self._prev
        self._prev = self._pending
        self.n_steps += 1
        self.trace.append(1 if is_phantom else 0)

        result = None
        if is_phantom:
            self.n_phantom += 1
            if not self.skip_phantom:
                self.n_added += 1
                result = self._orig_add(obs, next_obs, action, reward, done, infos)
        else:
            self.n_added += 1
            result = self._orig_add(obs, next_obs, action, reward, done, infos)

        if self.on_step is not None:
            self.on_step(self.n_steps)
        return result

    def restore(self) -> None:
        self.dqn.envs.step = self._orig_step
        self.dqn.rb.add = self._orig_add

    @property
    def frac_poison(self) -> float:
        """Share of environment steps that were phantom transitions.

        Identity worth knowing: this tracks 1 / E[episode length], so it can be
        estimated from an episode CSV alone when no probe was attached.
        """
        return self.n_phantom / max(self.n_steps, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "terminal_adjustment_active": self.skip_phantom,
            "steps": self.n_steps,
            "transitions_stored": self.n_added,
            "phantoms_seen": self.n_phantom,
            "phantoms_stored": 0 if self.skip_phantom else self.n_phantom,
            "frac_poison": round(self.frac_poison, 5),
        }


# ---------------------------------------------------------------------------
# Greedy evaluation hook
#
# Training return under epsilon-greedy conflates policy quality with
# exploration noise, and DQN on CartPole decays from its peak even when
# healthy (deadly triad + epsilon pinned at end_e). A last-N-episodes training
# statistic therefore reports failure on runs that reached 200-400 mid-training.
# This hook periodically measures the greedy policy (epsilon = 0) on a separate
# environment and logs it to its own CSV. Treat it as the primary metric.
# ---------------------------------------------------------------------------
@dataclass
class GreedyEvalConfig:
    every_steps: int = 10_000
    n_episodes: int = 5
    env_id: str = "CartPole-v1"
    seed_offset: int = 10_000
    enabled: bool = True
    # optional zero-argument factory for the evaluation environment (e.g. an observation-selecting
    # wrapper). None (the default) keeps gym.make(env_id), i.e. the behaviour of every earlier experiment.
    env_factory: Optional[Any] = None


def evaluate_greedy(dqn: Any, cfg: GreedyEvalConfig, seed: int) -> float:
    """Mean return of the greedy policy over cfg.n_episodes fresh episodes."""
    import torch

    env = cfg.env_factory() if cfg.env_factory is not None else gym.make(cfg.env_id)
    returns = []
    try:
        net = dqn.q_network
        was_training = net.training
        net.eval()
        with torch.no_grad():
            for k in range(cfg.n_episodes):
                obs, _ = env.reset(seed=seed + cfg.seed_offset + k)
                done, total = False, 0.0
                while not done:
                    q = net(torch.as_tensor(np.asarray(obs, dtype=np.float32))
                            .unsqueeze(0).to(dqn.device))
                    action = int(torch.argmax(q, dim=1).item())
                    obs, reward, term, trunc, _ = env.step(action)
                    total += float(reward)
                    done = bool(term) or bool(trunc)
                returns.append(total)
        if was_training:
            net.train()
    finally:
        env.close()
    return float(np.mean(returns)) if returns else float("nan")


# ---------------------------------------------------------------------------
# SafeDQN
# ---------------------------------------------------------------------------
@dataclass
class TrainOutcome:
    run_name: str
    seed: int
    total_timesteps: int
    wall_seconds: float
    probe: Dict[str, Any]
    episodes_csv: str
    eval_csv: Optional[str]
    trace_npy: Optional[str]
    extra: Dict[str, Any] = field(default_factory=dict)


class SafeDQN:
    """Thin orchestration around simplyqrl.dqn.DQN.

    Not a subclass on purpose: upstream's train() is monolithic, so we attach
    behaviour through the probe (which fires once per environment step) instead
    of reimplementing the loop. Less code of ours to be wrong, and the upstream
    algorithm stays exactly upstream's.
    """

    def __init__(
        self,
        agent_type: str,
        agent_config: Dict[str, Any],
        run_name: str,
        seed: int = 1,
        env_id: str = "CartPole-v1",
        terminal_adjustment: bool = True,
        eval_cfg: Optional[GreedyEvalConfig] = None,
        outdir: Optional[pathlib.Path] = None,
        post_build: Optional[Any] = None,
        env_factory: Optional[Any] = None,
        **dqn_kwargs: Any,
    ):
        # zero-argument factory for the training environment (e.g. make_selected_env). None (the
        # default) keeps gym.make(env_id) as before. upstream's make_vec_env accepts a callable.
        self.env_factory = env_factory
        # post_build(dqn) -> hook | None. Called once, after DQN() is constructed and before train(); it may
        # replace the optimiser or the networks (core/lr_equivalence.py) and returns an object with on_step(n) and
        # write(path). None (the default) leaves the pipeline exactly as it was.
        self.post_build = post_build
        self.agent_type = agent_type
        self.agent_config = agent_config
        self.run_name = run_name
        self.seed = seed
        self.env_id = env_id
        self.terminal_adjustment = terminal_adjustment
        self.eval_cfg = eval_cfg or GreedyEvalConfig()
        # Resolved to absolute at construction time, deliberately: train()
        # os.chdir()s INTO self.outdir for upstream's relative runs/*.csv
        # writes, then back out in a finally. Any self.outdir-relative path
        # built INSIDE that window (e.g. the weights_path below) would
        # otherwise double-resolve against the now-changed cwd (a relative output directory used to crash a smoke run).
        self.outdir = (pathlib.Path(outdir) if outdir else pathlib.Path.cwd()).resolve()
        self.dqn_kwargs = dqn_kwargs

    def train(self, total_timesteps: int, progress_bar: bool = False) -> TrainOutcome:
        from simplyqrl.dqn import DQN

        self.outdir.mkdir(parents=True, exist_ok=True)
        eval_rows: List[tuple] = []
        cwd = os.getcwd()
        t0 = time.time()
        try:
            # upstream writes ./runs/{run_name}.csv relative to cwd, flushed
            # once per episode - which is what makes partial results survive a
            # Colab disconnect.
            os.chdir(self.outdir)
            dqn = DQN(
                env=self.env_factory if self.env_factory is not None else gym.make(self.env_id),
                agent_type=self.agent_type,
                agent_config=self.agent_config,
                run_name=self.run_name,
                seed=self.seed,
                **self.dqn_kwargs,
            )
            diag = self.post_build(dqn) if self.post_build is not None else None
            probe = AutoresetProbe(dqn, skip_phantom=self.terminal_adjustment)

            if self.eval_cfg.enabled:
                every = self.eval_cfg.every_steps

                def _hook(step_index: int) -> None:
                    if step_index % every:
                        return
                    score = evaluate_greedy(dqn, self.eval_cfg, self.seed)
                    eval_rows.append((step_index, score))

                probe.on_step = _hook

            if diag is not None:
                _prev_hook = probe.on_step

                def _with_diag(step_index: int, _prev=_prev_hook, _d=diag) -> None:
                    _d.on_step(step_index)
                    if _prev is not None:
                        _prev(step_index)

                probe.on_step = _with_diag

            dqn.train(total_timesteps=total_timesteps, progress_bar=progress_bar)

            # [weight saving] A single write, after training finishes, of the
            # ONLINE network's state_dict only - no optimiser state, no replay
            # buffer, no exploration-schedule position, no RNG streams. This is
            # NOT resumption support, which stays deliberately unsupported (see
            # docs/reproducibility.md):
            # a saved state_dict lets a FINISHED policy be inspected or reused
            # for inference (e.g. extracting a trained circuit's Fourier
            # spectrum), it does not let
            # training be continued from this point. A few KB per cell.
            import torch
            weights_path = self.outdir / f"{self.run_name}_weights.pt"
            torch.save(dqn.q_network.state_dict(), weights_path)

            probe.restore()
            with contextlib.suppress(Exception):
                dqn.close()
        finally:
            os.chdir(cwd)

        wall = time.time() - t0
        trace_path = self.outdir / f"{self.run_name}_trace.npy"
        np.save(trace_path, np.asarray(probe.trace, dtype=np.uint8))

        eval_path = None
        if eval_rows:
            eval_path = self.outdir / "runs" / f"{self.run_name}_eval.csv"
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            with open(eval_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["global_step", "greedy_return"])
                w.writerows(eval_rows)

        return TrainOutcome(
            run_name=self.run_name,
            seed=self.seed,
            total_timesteps=total_timesteps,
            wall_seconds=round(wall, 1),
            probe=probe.summary(),
            episodes_csv=str(self.outdir / "runs" / f"{self.run_name}.csv"),
            eval_csv=str(eval_path) if eval_path else None,
            trace_npy=str(trace_path),
            # Optional by construction (TrainOutcome.extra already defaults to
            # {}), so manifests written before this change keep parsing fine -
            # they simply have no "weights_path" key. See docs/adjustments.md
            # for the compatibility note.
            extra=dict({"weights_path": str(weights_path)},
                       **({"diag_csv": diag.write(self.outdir / "runs" / f"{self.run_name}_diag.csv")}
                          if diag is not None else {})),
        )
