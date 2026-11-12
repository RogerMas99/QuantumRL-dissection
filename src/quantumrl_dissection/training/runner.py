"""
Resumable run orchestration.

Colab sessions die. Long PQC runs die with them. This module makes a grid
restartable: every finished run drops a manifest, and re-running the grid skips
what is already on disk. Episode CSVs are written by upstream with a flush per
episode, so even an interrupted run leaves a usable partial learning curve.

Convention: the raw artefacts of a run (manifest, episode and evaluation CSV, TensorBoard logs) are written to the
output directory given to the runner, normally outside the repository; ``results/`` holds the derived data.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import time
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from quantumrl_dissection.arms import build_arm_config
from quantumrl_dissection.training.dqn import GreedyEvalConfig, SafeDQN


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=pathlib.Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


@dataclass
class RunSpec:
    """One cell of an experiment grid."""
    arm: str
    seed: int
    terminal_adjustment: bool
    total_timesteps: int = 60_000
    dqn_kwargs: Dict[str, Any] = field(default_factory=dict)
    tag: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)      # overrides of the arm's configuration (diagnostics, ...)

    @property
    def run_name(self) -> str:
        """``<arm>[_no_terminal_adjustment]__s<seed>[__<tag>]``: the name of the arm describes the specification."""
        arm = self.arm if self.terminal_adjustment else f"{self.arm}_no_terminal_adjustment"
        base = f"{arm}__s{self.seed}"
        return f"{base}__{self.tag}" if self.tag else base



# Fields that make two runs the same run. If any of these differ, an existing
# manifest is NOT an answer to the question being asked, and reusing it silently
# would report a 1,500-step smoke run as a 100,000-step result - which is exactly
# what happened before this guard existed.
REUSE_KEYS = ("arm", "seed", "terminal_adjustment", "total_timesteps", "dqn_kwargs", "tag", "overrides")


def reuse_or_none(manifest_path, expected, env_id=None, on_mismatch="error"):
    """Return an existing manifest only if it answers the same question.

    `run_name` deliberately encodes only arm, terminal-adjustment state, seed and tag - it is
    meant to be readable. That makes it an incomplete key: two runs with the same
    name can differ in step budget, batch size, buffer size or environment. This
    function closes that gap by comparing the stored spec against the requested
    one before agreeing to skip.

    on_mismatch:
        "error"  raise, naming the fields that differ (default - loud beats wrong)
        "rerun"  ignore the stale manifest and recompute
        "legacy" accept manifests that simply predate a field, but still refuse
                 ones that actively contradict the request
        "skip"   reuse anyway; only for deliberate archaeology
    """
    manifest_path = pathlib.Path(manifest_path)
    if not manifest_path.exists():
        return None
    stored = json.loads(manifest_path.read_text())
    stored_spec = stored.get("spec", stored)

    outcome = stored.get("outcome", {})
    diffs, unverifiable = [], []
    for key in REUSE_KEYS:
        if key not in expected:
            continue
        if key in stored_spec:
            found = stored_spec[key]
        elif key == "total_timesteps" and "total_timesteps" in outcome:
            # Manifests written before the spec was recorded in full still carry
            # the step budget inside the outcome, so this one is recoverable.
            found = outcome["total_timesteps"]
        else:
            unverifiable.append(key)
            continue
        if found != expected[key]:
            diffs.append(f"    {key}: on disk {found!r} != requested {expected[key]!r}")
    if env_id is not None and stored.get("env_id") not in (None, env_id):
        diffs.append(f"    env_id: on disk {stored.get('env_id')!r} != requested {env_id!r}")

    if diffs:
        if on_mismatch == "skip":
            return stored
        if on_mismatch == "rerun":
            return None
        raise RuntimeError(
            f"{manifest_path.name} exists but was produced by a different run:\n"
            + "\n".join(diffs)
            + "\n  Reusing it would report one experiment's numbers as another's."
            "\n  Fix by one of: delete that manifest, write to a different --outdir,"
            "\n  give the run a distinct tag, or pass force=True to recompute."
        )

    if unverifiable:
        # A manifest written before these fields were recorded. Absence is not evidence of a mismatch, so refusing
        # outright would throw away real compute - but accepting silently could report one run's numbers as another's.
        if on_mismatch in ("skip", "legacy"):
            return stored
        if on_mismatch == "rerun":
            return None
        raise RuntimeError(
            f"{manifest_path.name} does not record {', '.join(unverifiable)}, so it cannot be checked against this request."
            " Pass on_mismatch='legacy' to accept it as it is, or delete it to recompute."
        )

    return stored


# ---------------------------------------------------------------------------
# Cooperative claiming, for running the same grid from several Colab sessions.
#
# The alternative is partitioning by hand - one session per seed range - which
# works and needs no code. This exists because with cells that run for hours,
# hand-partitioning means deciding the split up front and living with it: if one
# session dies at 3am its slice simply does not get done, and if one slice turns
# out cheaper that session sits idle.
#
# Best-effort, and honestly so. Google Drive's FUSE mount gives no atomic
# create, so two sessions starting the same cell within the sync window can both
# claim it. The failure mode is duplicated compute, never a corrupted result:
# both write the same manifest for the same spec. With multi-hour cells the
# window is a rounding error.
#
# Stale locks are reclaimed after `ttl_hours` - a disconnected session must not
# reserve a cell forever.
# ---------------------------------------------------------------------------

def _lock_path(manifest_path) -> pathlib.Path:
    return pathlib.Path(str(manifest_path).replace(".manifest.json", ".lock.json"))


def claim_cell(manifest_path, ttl_hours: float = 12.0, owner: str = "") -> bool:
    """Try to reserve a cell. False means another live session holds it."""
    lock = _lock_path(manifest_path)
    now = time.time()
    if lock.exists():
        try:
            held = json.loads(lock.read_text())
            age_h = (now - float(held.get("t", 0))) / 3600.0
            if age_h < ttl_hours:
                return False
        except Exception:
            pass                    # unreadable lock: treat as stale
    owner = owner or f"{platform.node()}:{os.getpid()}"
    try:
        lock.write_text(json.dumps({"owner": owner, "t": now}))
    except OSError:
        return True                 # cannot lock (read-only fs) - just run
    time.sleep(0.5)                 # let a competing write land before checking
    try:
        return json.loads(lock.read_text()).get("owner") == owner
    except Exception:
        return True


def release_cell(manifest_path) -> None:
    """Drop the reservation. Safe to call when no lock is held."""
    try:
        _lock_path(manifest_path).unlink()
    except OSError:
        pass


def run_arm(
    spec: RunSpec,
    outdir: pathlib.Path,
    env_id: str = "CartPole-v1",
    eval_cfg: Optional[GreedyEvalConfig] = None,
    progress_bar: bool = False,
    force: bool = False,
    on_mismatch: str = "error",
    claim: bool = False,
    claim_ttl_hours: float = 12.0,
) -> Dict[str, Any]:
    """Execute one cell, or return the existing manifest if it answers the same
    question. A manifest with the same name but a different spec is a mismatch,
    not a hit - see `reuse_or_none`."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / f"{spec.run_name}.manifest.json"

    if not force:
        hit = reuse_or_none(manifest_path, asdict(spec), env_id=env_id,
                            on_mismatch=on_mismatch)
        if hit is not None:
            return hit

    if claim and not claim_cell(manifest_path, claim_ttl_hours):
        return {"run_name": spec.run_name, "claimed_elsewhere": True}

    agent_type, agent_config = build_arm_config(spec.arm, env_id=env_id, **spec.overrides)
    from quantumrl_dissection.training.lr_equivalence import make_post_build    # None for every arm that has no lr_* key

    # An arm may select components of the observation in the environment (``_env_select``): the vendored library applies
    # ``reuse_indices`` only when the observation is replicated.
    env_factory = None
    select = agent_config.get("_env_select")
    if select is not None:
        import dataclasses
        from quantumrl_dissection.environments import make_selected_env
        env_factory = make_selected_env(env_id, select)
        eval_cfg = dataclasses.replace(eval_cfg or GreedyEvalConfig(env_id=env_id), env_factory=env_factory)

    runner = SafeDQN(
        agent_type=agent_type,
        agent_config=agent_config,
        run_name=spec.run_name,
        seed=spec.seed,
        env_id=env_id,
        terminal_adjustment=spec.terminal_adjustment,
        eval_cfg=eval_cfg,
        outdir=outdir,
        post_build=make_post_build(agent_config),
        env_factory=env_factory,
        **spec.dqn_kwargs,
    )
    outcome = runner.train(spec.total_timesteps, progress_bar=progress_bar)

    manifest = {
        "spec": asdict(spec),
        "run_name": spec.run_name,
        "outcome": asdict(outcome),
        "agent_type": agent_type,
        "agent_config": {k: str(v) for k, v in agent_config.items()},
        "env_id": env_id,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
    }
    if select is not None:
        manifest["observation_selection"] = {"where": "environment", "components": list(select)}
    from quantumrl_dissection.training.provenance import provenance_block
    prov = provenance_block(agent_type)
    if prov is not None:
        # Simulator result: stamped so that no table can pick it up unnoticed.
        manifest["provenance"] = prov
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    if claim:
        release_cell(manifest_path)
    return manifest


def run_grid(
    specs: Iterable[RunSpec],
    outdir: pathlib.Path,
    env_id: str = "CartPole-v1",
    eval_cfg: Optional[GreedyEvalConfig] = None,
    progress_bar: bool = False,
    verbose: bool = True,
    claim: bool = False,
    claim_ttl_hours: float = 12.0,
) -> List[Dict[str, Any]]:
    results = []
    specs = list(specs)
    for i, spec in enumerate(specs, 1):
        if verbose:
            print(f"[{i}/{len(specs)}] {spec.run_name}", flush=True)
        try:
            m = run_arm(spec, outdir, env_id=env_id, eval_cfg=eval_cfg,
                        progress_bar=progress_bar, claim=claim,
                        claim_ttl_hours=claim_ttl_hours)
            results.append(m)
            if m.get("claimed_elsewhere"):
                if verbose:
                    print("    busy - another session holds this cell", flush=True)
                continue
            if verbose:
                p = m["outcome"]["probe"]
                print(f"    ok  {m['outcome']['wall_seconds']}s  "
                      f"phantoms {100 * p['frac_poison']:.2f}% of steps  "
                      f"terminal adjustment {'on' if p['terminal_adjustment_active'] else 'off'}",
                      flush=True)
        except Exception as exc:  # keep the grid alive; report at the end
            if verbose:
                print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"run_name": spec.run_name, "error": repr(exc)})
    return results
