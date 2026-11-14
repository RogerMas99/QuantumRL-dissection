"""Training: adjustments to SimplyQRL, DQN with the terminal adjustment and greedy evaluation, and the run of one cell."""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import inspect
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch.nn as nn

from quantumrl_dissection.arms import build_arm_config
from quantumrl_dissection.models import make_selected_env


_APPLIED: Dict[str, str] = {}

# --- Adjustments to SimplyQRL. They rebind attributes of the loaded library when the package is imported; each one first checks
# --- that the library still has the shape it expects, so that a change upstream fails loudly instead of becoming a silent no-op.

_AGENT_TYPE_ALIASES = {
    "classic": "mlp", "classical": "mlp", "mlp": "mlp",
    "hybrid": "hybrid", "quantum": "hybrid",
}


def _rebind_build_agent(build_agent) -> None:
    """``simplyqrl.dqn`` and ``simplyqrl.ppo`` import ``build_agent`` by name, so rebinding it on ``agents`` alone does not reach them."""
    from simplyqrl import agents as up

    up.build_agent = build_agent
    for mod_name in ("simplyqrl.dqn", "simplyqrl.ppo"):
        try:
            mod = __import__(mod_name, fromlist=["build_agent"])
        except Exception:  # pragma: no cover - optional module
            continue
        if hasattr(mod, "build_agent"):
            mod.build_agent = build_agent


def _patch_agent_type_aliases() -> None:
    """``build_agent`` returns None for any agent type other than "mlp" and "hybrid"; accept "classic" and raise on unknown names."""
    from simplyqrl import agents as up

    original = up.build_agent
    if getattr(original, "_quantumrl_dissection_patched", False):
        return
    src = inspect.getsource(original)
    if 'if type == "mlp"' not in src or 'elif type == "hybrid"' not in src:
        raise RuntimeError('upstream build_agent no longer dispatches on "mlp" and "hybrid" as expected')

    def build_agent(type, obs_shape, n_actions, config, is_qnet=False):  # noqa: A002
        key = str(type).lower()
        if key not in _AGENT_TYPE_ALIASES:
            raise ValueError("unknown agent_type %r. Known: %s" % (type, sorted(set(_AGENT_TYPE_ALIASES))))
        return original(_AGENT_TYPE_ALIASES[key], obs_shape, n_actions, config, is_qnet=is_qnet)

    build_agent._quantumrl_dissection_patched = True
    _rebind_build_agent(build_agent)
    _APPLIED["agent_type_aliases"] = 'agent_type aliases: "classic"/"classical" -> "mlp"'


def _patch_output_scale() -> None:
    """In the Q-network branch of ``HybridAgent`` the output scale is appended to a list that ``nn.Sequential`` has already consumed."""
    from simplyqrl import agents as up

    if getattr(up.HybridAgent, "_quantumrl_dissection_output_scale_patched", False):
        return
    src = inspect.getsource(up.HybridAgent.__init__)
    if "use_output_scaling" not in src or "OutputScale" not in src:
        raise RuntimeError("upstream HybridAgent no longer mentions use_output_scaling and OutputScale")
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
            return
        init_val = cfg.get("output_scale_init", 2.0)
        self.network = nn.Sequential(*list(net), up.OutputScale(act_dim, init_val))

    up.HybridAgent.__init__ = __init__
    up.HybridAgent._quantumrl_dissection_output_scale_patched = True
    _APPLIED["output_scale"] = "OutputScale now actually appended to HybridAgent.network"


_NEW_AGENT_TYPES = {"su2", "sv", "fourier_additive", "linear_on_bits", "classical_twin", "bits_mlp"}


def _patch_new_agent_types() -> None:
    """Add the agent types of this package (simulators, classical twin, ceilings); every other name goes to the previous dispatch."""
    from simplyqrl import agents as up

    previous = up.build_agent
    if getattr(previous, "_quantumrl_dissection_new_types_patched", False):
        return

    def build_agent(type, obs_shape, n_actions, config, is_qnet=False):  # noqa: A002
        key = str(type).lower()
        if key in _NEW_AGENT_TYPES:
            if not is_qnet:
                raise ValueError(f"agent_type={key!r} only supports is_qnet=True (DQN); it has no actor-critic construction.")
            cfg = config or {}
            obs_dim = int(np.prod(obs_shape))
            if key == "su2":
                from quantumrl_dissection.simulators import SU2HybridAgent
                return SU2HybridAgent(obs_dim, n_actions, cfg, is_qnet=True)
            if key == "sv":
                from quantumrl_dissection.simulators import SVHybridAgent
                return SVHybridAgent(obs_dim, n_actions, cfg, is_qnet=True)
            if key == "bits_mlp":
                from quantumrl_dissection.models import BitsMLPAgent
                return BitsMLPAgent(obs_dim, n_actions, cfg, is_qnet=True)
            if key == "classical_twin":
                from quantumrl_dissection.models import ClassicalTwinAgent
                return ClassicalTwinAgent(obs_dim, n_actions, cfg, is_qnet=True)
            if key == "fourier_additive":
                from quantumrl_dissection.models import FourierAdditiveCeiling
                return FourierAdditiveCeiling(
                    n_qubits=cfg.get("n_qubits", 4), n_layers=cfg.get("n_layers_q", 1), n_actions=n_actions,
                    emb_indices=cfg.get("emb_indices", None), transform_fn=cfg.get("transform_fn", None),
                    circ_type=cfg.get("circ_type", "skolik"), n_data=obs_dim)
            from quantumrl_dissection.models import linear_on_bits_ceiling
            return linear_on_bits_ceiling(cfg.get("n_qubits", 4), n_actions, transform_fn=cfg.get("transform_fn", None))
        return previous(type, obs_shape, n_actions, config, is_qnet=is_qnet)

    build_agent._quantumrl_dissection_new_types_patched = True
    _rebind_build_agent(build_agent)
    _APPLIED["simulator_and_ceiling_agent_types"] = "agent_type dispatch extended: su2, sv, classical_twin, fourier_additive, linear_on_bits, bits_mlp"


def apply_upstream_patches() -> None:
    """Apply the adjustments; idempotent."""
    _patch_agent_type_aliases()
    _patch_output_scale()
    _patch_new_agent_types()


def upstream_report() -> Dict[str, Any]:
    """The adjustments applied and the versions of the libraries they were applied to."""
    import gymnasium
    import pennylane
    import simplyqrl

    return {"patches_applied": dict(_APPLIED), "simplyqrl": getattr(simplyqrl, "__file__", "?"),
            "gymnasium": gymnasium.__version__, "pennylane": pennylane.__version__}


# --- DQN with the terminal adjustment ----------------------------------------------------------------------------------------------
# Since gymnasium 1.0 the vectorised environments autoreset at the step after a termination: that step ignores the action and
# returns reward 0 with the reset observation. The library stores it in the replay buffer, which teaches Q(terminal) ~ gamma V(s0).
# The probe recognises that transition (the add that follows a step that ended an episode) and either skips it (the terminal
# adjustment) or lets it through; in both cases it counts it.


class AutoresetProbe:
    """Wraps envs.step and rb.add on a constructed DQN."""

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
        self.on_step = None

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
        """Share of environment steps that were phantom transitions."""
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


@dataclass
class GreedyEvalConfig:
    every_steps: int = 10_000
    n_episodes: int = 5
    env_id: str = "CartPole-v1"
    seed_offset: int = 10_000
    enabled: bool = True
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


@dataclass
class TrainOutcome:
    run_name: str
    seed: int
    total_timesteps: int
    wall_seconds: float
    probe: Dict[str, Any]
    episodes_csv: str
    eval_csv: Optional[str]


class SafeDQN:
    """Thin orchestration around simplyqrl.dqn.DQN: the probe, the greedy evaluation and the hook of the learning-rate variants."""

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
        self.env_factory = env_factory
        self.post_build = post_build
        self.agent_type = agent_type
        self.agent_config = agent_config
        self.run_name = run_name
        self.seed = seed
        self.env_id = env_id
        self.terminal_adjustment = terminal_adjustment
        self.eval_cfg = eval_cfg or GreedyEvalConfig()
        # Absolute: ``train`` changes into this directory, because the library writes runs/<run_name>.csv relative to it.
        self.outdir = (pathlib.Path(outdir) if outdir else pathlib.Path.cwd()).resolve()
        self.dqn_kwargs = dqn_kwargs

    def train(self, total_timesteps: int, progress_bar: bool = False) -> TrainOutcome:
        from simplyqrl.dqn import DQN

        self.outdir.mkdir(parents=True, exist_ok=True)
        eval_rows: List[tuple] = []
        cwd = os.getcwd()
        t0 = time.time()
        try:
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

            self.dqn, self.hook = dqn, diag

            probe.restore()
            with contextlib.suppress(Exception):
                dqn.close()
        finally:
            os.chdir(cwd)

        wall = time.time() - t0

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
        )


@dataclass
class RunSpec:
    """One cell of an experiment: an arm, a seed and the protocol."""
    arm: str
    seed: int
    terminal_adjustment: bool = True
    total_timesteps: int = 100_000
    dqn_kwargs: Dict[str, Any] = field(default_factory=dict)
    tag: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        arm = self.arm if self.terminal_adjustment else f"{self.arm}_no_terminal_adjustment"
        base = f"{arm}__s{self.seed}"
        return f"{base}__{self.tag}" if self.tag else base


def run_cell(spec: RunSpec, outdir: pathlib.Path, env_id: str = "CartPole-v1", eval_cfg: Optional[GreedyEvalConfig] = None,
             progress_bar: bool = False) -> TrainOutcome:
    """Train one cell and write its episode and greedy-evaluation CSVs under ``outdir/runs``."""
    from quantumrl_dissection.learning_rate import make_post_build

    agent_type, agent_config = build_arm_config(spec.arm, env_id=env_id, **spec.overrides)
    env_factory = None
    select = agent_config.get("_env_select")
    if select is not None:
        env_factory = make_selected_env(env_id, select)
        eval_cfg = dataclasses.replace(eval_cfg or GreedyEvalConfig(env_id=env_id), env_factory=env_factory)
    runner = SafeDQN(agent_type=agent_type, agent_config=agent_config, run_name=spec.run_name, seed=spec.seed, env_id=env_id,
                     terminal_adjustment=spec.terminal_adjustment, eval_cfg=eval_cfg, outdir=outdir,
                     post_build=make_post_build(agent_config), env_factory=env_factory, **spec.dqn_kwargs)
    return runner.train(spec.total_timesteps, progress_bar=progress_bar)
