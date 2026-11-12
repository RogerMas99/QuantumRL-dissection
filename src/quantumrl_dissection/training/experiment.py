"""Experiment configurations (``configs/*.toml``): parse a file into the cells it runs and run them.

A configuration holds one or more stages. A stage runs a list of arms on a list of seeds with one protocol:

    algorithm            "dqn" or "ppo"
    environment          "cartpole" or "frozenlake" (the one-hot reference arm is routed to its own id)
    arms, seeds          arm names (``arms.py``) and seeds ("1-10", "11-100", "1-10,21-30")
    steps, eval_every, eval_episodes
    dqn                  DQN hyperparameters that differ from the library defaults
    terminal_adjustment  false runs the arm with the adjustment disabled (the run name gets ``_no_terminal_adjustment``)
    diagnostics          learning-rate experiment: attach the diagnostics (and seed the exploration sampler)
    arm_overrides        per-arm overrides of the configuration of the arm
"""
from __future__ import annotations

import pathlib
import tomllib
from typing import Any, Dict, List, Optional, Tuple

from quantumrl_dissection.arms import ENVIRONMENTS, environment_of


def parse_seeds(text) -> List[int]:
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_config(path) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def stage_cells(stage: Dict[str, Any], seeds: Optional[List[int]] = None, arms: Optional[List[str]] = None
                ) -> List[Tuple[str, Any]]:
    """The cells of a stage as ``(environment id, RunSpec | PPOSpec)``, in a deterministic order (seed, arm)."""
    from quantumrl_dissection.training.runner import RunSpec
    from quantumrl_dissection.training.ppo import PPOSpec
    sd = [s for s in parse_seeds(stage["seeds"]) if seeds is None or s in seeds]
    ar = [a for a in stage["arms"] if arms is None or a in arms]
    cells: List[Tuple[str, Any]] = []
    for s in sd:
        for a in ar:
            if stage["algorithm"] == "ppo":
                cells.append((ENVIRONMENTS["cartpole"]["env_id"], PPOSpec(arm=a, seed=s, total_timesteps=int(stage["steps"]))))
                continue
            overrides: Dict[str, Any] = {}
            if stage.get("diagnostics"):
                overrides["lr_diag"] = True
            overrides.update(stage.get("arm_overrides", {}).get(a, {}))
            cells.append((environment_of(a), RunSpec(
                arm=a, seed=s, terminal_adjustment=bool(stage.get("terminal_adjustment", True)),
                total_timesteps=int(stage["steps"]), dqn_kwargs=dict(stage.get("dqn", {})), tag=stage.get("tag", ""),
                overrides=overrides)))
    return cells


def config_cells(cfg: Dict[str, Any], stage: Optional[str] = None, seeds: Optional[List[int]] = None,
                 arms: Optional[List[str]] = None):
    for st in cfg["stage"]:
        if stage and st["name"] != stage:
            continue
        yield st, stage_cells(st, seeds, arms)


def run_stage(stage: Dict[str, Any], cells, outdir: pathlib.Path, shard: Optional[Tuple[int, int]] = None,
              claim: bool = False, progress_bar: bool = False) -> None:
    """Run the cells of a stage into ``outdir`` (resumable: finished cells are skipped)."""
    if shard:
        i, n = shard
        cells = [c for k, c in enumerate(cells) if k % n == i]
    if stage["algorithm"] == "ppo":
        from quantumrl_dissection.training.ppo import run_ppo_cell
        for k, (_, spec) in enumerate(cells, 1):
            print(f"[{k}/{len(cells)}] {spec.run_name}", flush=True)
            run_ppo_cell(spec, outdir, eval_episodes=int(stage.get("eval_episodes", 20)))
        return
    from quantumrl_dissection.training.dqn import GreedyEvalConfig
    from quantumrl_dissection.training.runner import run_grid
    by_env: Dict[str, list] = {}
    for env_id, spec in cells:
        by_env.setdefault(env_id, []).append(spec)
    for env_id, specs in by_env.items():
        eval_cfg = GreedyEvalConfig(env_id=env_id, every_steps=int(stage.get("eval_every", 10_000)),
                                    n_episodes=int(stage.get("eval_episodes", 5)))
        run_grid(specs, outdir, env_id=env_id, eval_cfg=eval_cfg, progress_bar=progress_bar, claim=claim)
