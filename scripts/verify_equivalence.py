"""Equivalence of the training code of this repository with the code that produced the runs.

The code of the original repository is exported with ``git archive`` at the revision that produced each run (``effective_revision`` of
``results/runs.csv`` when it is set, otherwise ``code_revision``), and both codes are run in the same process settings: double precision,
one thread, the random actions of the exploration drawn from a generator that the harness seeds. Two phases:

``params``  for every arm of ``results/runs.csv`` that trains with DQN and every revision at which it was run: the number of trainable
            parameters and the name and shape of each tensor of the online network are the same in both codes.
``traces``  for representative arms of every type, at the first and at the last revision at which they were run: the actions taken
            during ``--steps`` environment steps (with updates of the network from step ``learning_starts``) are the same, and the weights
            of the online network at the end differ by less than ``--tol`` (1e-12).
``full``    one arm in full: 100,000 steps with the protocol of its configuration.

    python scripts/verify_equivalence.py --old-repo <path to the original repository> --workdir <scratch folder> params
    python scripts/verify_equivalence.py --old-repo ... --workdir ... traces --steps 2000
    python scripts/verify_equivalence.py --old-repo ... --workdir ... full --arm reference_control_reuse8 --config rq3_output_reuse --stage reuse_sweep

The report is written to ``<workdir>/report_<phase>.json``. A difference is reported, never adjusted.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIRST_WITH_LIBRARY = "18e124a"          # first revision of the original repository that contains the library
EARLY = "41005cc"                       # revision used for runs without a recorded revision that started before the library was added
START_LIBRARY_UTC = "2026-08-05 21:42:00"

# Arms of every type that the traces cover (config, stage, arm). Types: real circuit with and without entanglement, depth, reuse, the arms
# ported from the reference study, FrozenLake circuits, simulators, and the classical arms.
TRACE_ARMS = [
    ("rq2_capacity", "capacity_ladder", "hybrid"),
    ("rq2_capacity", "capacity_ladder", "reference_control"),
    ("rq2_capacity", "capacity_ladder", "matched_control"),
    ("rq2_capacity", "capacity_ladder", "large_network"),
    ("rq2_capacity", "classical_twin", "classical_twin_reuse4"),
    ("rq2_capacity", "classical_twin", "classical_twin_unbounded_reuse4"),
    ("rq3_output_reuse", "reuse_sweep", "hybrid_reuse4"),
    ("rq3_output_reuse", "reuse_sweep", "reference_control_reuse8"),
    ("rq3_output_reuse", "reuse_sweep", "matched_control_reuse4"),
    ("rq3_output_reuse", "replicated_control", "matched_control_replicated8"),
    ("rq3_output_reuse", "reference_control_without_reuse", "reference_control_reuse1"),
    ("rq3_depth", "depth_sweep", "hybrid_depth1"),
    ("rq3_depth", "depth_reuse_crossing", "hybrid_depth2_reuse4"),
    ("rq3_entanglement", "depth_sweep_unentangled", "hybrid_unentangled"),
    ("rq3_entanglement", "reuse_sweep_unentangled", "hybrid_unentangled_reuse4"),
    ("rq1_porting", "porting", "porting_skolik_4q_depth2"),
    ("rq1_porting", "porting", "porting_skolik_4q_depth2_unentangled"),
    ("rq1_porting", "porting", "porting_skolik_8q_depth1"),
    ("rq1_porting", "porting", "porting_salinas_1q_depth1"),
    ("rq1_porting", "porting", "porting_salinas_2q_depth2"),
    ("rq1_porting", "porting", "porting_hsiao_reuse4"),
    ("rq1_porting", "porting", "porting_hsiao_reuse4_entangled"),
    ("rq4_ceilings", "cartpole_ceiling", "fourier_ceiling"),
    ("rq4_ceilings", "frozenlake_ceilings", "frozenlake_scalar_ceiling_depth5"),
    ("rq4_ceilings", "frozenlake_ceilings", "frozenlake_binary_linear_ceiling"),
    ("rq5_frozenlake", "frozenlake_arms", "frozenlake_binary_depth1"),
    ("rq5_frozenlake", "frozenlake_arms", "frozenlake_scalar_depth5"),
    ("rq5_frozenlake", "frozenlake_arms", "frozenlake_binary_unentangled_depth1"),
    ("rq5_frozenlake", "frozenlake_arms", "frozenlake_onehot_reference"),
    ("rq5_frozenlake", "frozenlake_arms", "frozenlake_matched_control"),
    ("terminal_ablation", "capacity_ladder_no_terminal_adjustment", "reference_control"),
    ("lr_experiment", "dqn_twin", "classical_twin_head_lr_x4"),
    ("lr_experiment", "dqn_twin", "classical_twin_collapsed4"),
    ("lr_experiment", "dqn_twin", "classical_twin_grad_x8"),
    ("lr_experiment", "dqn_twin", "classical_twin_all_lr_x2"),
    ("lr_experiment", "ceiling_with_reuse", "fourier_ceiling"),
    ("lr_experiment", "ceiling_with_reuse", "fourier_ceiling_distinct_lr_x2_collapsed"),
    ("lr_experiment", "matched_control_with_reuse", "matched_control"),
    ("lr_experiment", "matched_control_with_reuse", "network_no_replication_w7_lr_x4_collapsed"),
    ("lr_experiment", "bits_networks", "frozenlake_bits_network_w6"),
    ("lr_experiment", "dqn_simulator_hybrid", "sim_statevector_hybrid_head_lr_x4"),
    ("lr_experiment", "dqn_simulator_hybrid", "sim_statevector_hybrid_collapsed4"),
    ("simulator_validation", "emulator_reference", "sim_emulator_hybrid_unentangled"),
    ("simulator_validation", "validation_battery_cartpole", "sim_statevector_hybrid"),
    ("simulator_validation", "validation_battery_cartpole", "sim_emulator_hybrid_unentangled_reuse4"),
    ("simulator_validation", "validation_battery_frozenlake_1", "sim_statevector_frozenlake_binary_depth1"),
    ("simulator_validation", "validation_battery_frozenlake_1", "sim_emulator_frozenlake_scalar_depth5"),
]

# Labels that the first experiment scripts built by calling a helper instead of a registry key.
EARLY_LABELS = {"hybrid_OR": "hybrid_or", "hybrid_DR": "hybrid_dr", "classical_OR": "classical_or"}


# ---------------------------------------------------------------------------------------------------------------------------------
# worker: runs in a process whose PYTHONPATH selects the old or the new code
# ---------------------------------------------------------------------------------------------------------------------------------
def _old_resolve(label: str, env_id: str):
    import re
    import qrl_dissection  # noqa: F401
    from qrl_dissection.core import configs as C
    m = re.fullmatch(r"hybrid_OR(\d+)", label)
    if m:
        return "hybrid", C.hybrid_or_config(int(m.group(1)))
    m = re.fullmatch(r"hybrid_DR(\d+)(_noent)?", label)
    if m:
        cfg = C.hybrid_dr_config(int(m.group(1)))
        if m.group(2):
            cfg["ent"] = False
        return "hybrid", cfg
    m = re.fullmatch(r"classical_OR(\d+)", label)
    if m:
        return "classic", C.paper_classical_or_config(int(m.group(1)))
    return C.build_arm_config(label, env_id=env_id)


def _worker(side: str, items_path: str, out_path: str) -> None:
    import numpy as np
    import torch
    import gymnasium as gym
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    holder = []
    from simplyqrl import dqn as sdqn
    _train = sdqn.DQN.train

    def to64(module, args):
        return tuple(x.double() if torch.is_tensor(x) and x.is_floating_point() else x for x in args)

    def train(self, *a, **k):
        holder.append(self)
        self._w0 = {k: v.detach().clone() for k, v in self.q_network.state_dict().items()}
        if torch.get_default_dtype() == torch.float64:      # the replay buffer returns float32 tensors: the networks take them in double precision
            for name in ("q_network", "target_network"):
                if hasattr(self, name):
                    getattr(self, name).register_forward_pre_hook(to64)
        return _train(self, *a, **k)
    sdqn.DQN.train = train

    items = json.load(open(items_path))
    results = {}
    for it in items:
        torch.set_default_dtype(torch.float64 if it.get("float64", True) else torch.float32)
        rng = np.random.default_rng(10_000 + int(it["seed"]))
        gym.spaces.Discrete.sample = lambda self, mask=None, probability=None, _r=rng: int(_r.integers(self.n)) + int(self.start)
        holder.clear()
        work = pathlib.Path(it["work"]) / side
        work.mkdir(parents=True, exist_ok=True)
        rec = {"id": it["id"]}
        try:
            kw = dict(it["kw"])
            if it.get("algo") == "ppo":
                work_dir = os.getcwd()
                os.chdir(work)
                try:
                    if side == "old":
                        import exp13_ppo as e13
                        from qrl_dissection.ppo.lr_ppo import LRPPO, make_selected_env
                        arm, cfg = e13.ARMS[it["old_label"]], e13.PAPER_PPO_CONFIG
                    else:
                        import quantumrl_dissection  # noqa: F401
                        from quantumrl_dissection.training.ppo import PPO_ARMS, REFERENCE_PPO_CONFIG, LRPPO, make_selected_env
                        arm, cfg = PPO_ARMS[it["arm"]], REFERENCE_PPO_CONFIG
                    ppo = LRPPO(env=make_selected_env("CartPole-v1"), seed=it["seed"], agent_type="mlp",
                                agent_config={"n_repeats": arm.R, "net_arch": []}, run_name="v",
                                head_lr_mult=float(arm.R) if arm.collapse else 1.0, collapse=arm.collapse, clip_mode=arm.clip_mode,
                                clip_R=arm.R, eval_every=10**9, eval_episodes=1, diag_every=10**9, float64=True, **cfg)
                    rec["n_params"] = int(sum(p.numel() for p in ppo.agent.parameters()))
                    ppo.train(total_timesteps=it["steps"])
                    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in ppo.agent.state_dict().items()}
                    rec["tensors"] = [[k, list(v.shape)] for k, v in sd.items()]
                    np.savez(pathlib.Path(out_path).with_suffix("").as_posix() + "__" + hashlib.sha1(it["id"].encode()).hexdigest()[:12] + ".npz",
                             actions=np.zeros(0, dtype=np.uint8), **{f"w{i}": v for i, v in enumerate(sd.values())})
                    rec["n_actions"] = 0
                finally:
                    os.chdir(work_dir)
                results[it["id"]] = rec
                json.dump(results, open(out_path, "w"), indent=1)
                continue
            if side == "old":
                from qrl_dissection.dqn import SafeDQN, GreedyEvalConfig
                agent_type, cfg = _old_resolve(it["old_label"], it["env_id"])
                post_build = None
                try:
                    from qrl_dissection.core.lr_equivalence import make_post_build
                    post_build = make_post_build(cfg)
                except Exception:
                    pass
                extra = {"post_build": post_build} if post_build is not None else {}
                if it["old_label"] == "classical_OR1":           # the reference control without reuse selects the observation in the environment
                    from qrl_dissection.ppo.lr_ppo import make_selected_env
                    extra["env_factory"] = make_selected_env(it["env_id"], (1, 2, 3))
                out = SafeDQN(agent_type=agent_type, agent_config=cfg, run_name=f"v__{abs(hash(it['id'])) % 10**8}", seed=it["seed"],
                              env_id=it["env_id"], fix_autoreset=it["terminal"], eval_cfg=GreedyEvalConfig(enabled=False),
                              outdir=work, **extra, **kw).train(it["steps"])
                trace = out.trace_npy
                rec["label"] = it["old_label"]
            else:
                import quantumrl_dissection  # noqa: F401
                from quantumrl_dissection.training.experiment import load_config, stage_cells
                from quantumrl_dissection.training.dqn import GreedyEvalConfig
                from quantumrl_dissection.training.runner import run_arm
                cfg = load_config(ROOT / "configs" / f"{it['config']}.toml")
                stage = next(s for s in cfg["stage"] if s["name"] == it["stage"])
                (env_id, spec), = stage_cells(stage, seeds=[it["seed"]], arms=[it["arm"]])
                spec.total_timesteps = it["steps"]
                spec.dqn_kwargs = dict(spec.dqn_kwargs, **kw)
                spec.tag = "v"
                m = run_arm(spec, work, env_id=env_id, eval_cfg=GreedyEvalConfig(env_id=env_id, enabled=False), force=True)
                trace = m["outcome"]["trace_npy"]
            dqn = holder[0]
            sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in dqn.q_network.state_dict().items()}
            rec["n_params"] = int(sum(p.numel() for p in dqn.q_network.parameters()))
            rec["tensors"] = [[k, list(v.shape)] for k, v in sd.items()]
            rec["moved"] = float(max((v.detach().cpu().numpy().astype(np.float64) - dqn._w0[k].cpu().numpy().astype(np.float64)).__abs__().max()
                                     for k, v in dqn.q_network.state_dict().items() if v.numel()))       # largest change of a weight during the run
            if it.get("trace", True):
                acts = np.load(trace)
                np.savez(pathlib.Path(out_path).with_suffix("") .as_posix() + "__" + hashlib.sha1(it["id"].encode()).hexdigest()[:12] + ".npz",
                         actions=acts, **{f"w{i}": v for i, v in enumerate(sd.values())})
                rec["n_actions"] = int(len(acts))
                rec["actions_sha"] = hashlib.sha1(acts.tobytes()).hexdigest()
        except Exception as e:                  # reported, never hidden
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        results[it["id"]] = rec
        json.dump(results, open(out_path, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------------------------------------------------------------
def _git(repo, *a, binary=False):
    r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True)
    return r.stdout if binary else r.stdout.decode().strip()


def export_old(repo, rev, base):
    d = pathlib.Path(base) / "old_code" / rev
    if not (d / "src").exists():
        d.mkdir(parents=True, exist_ok=True)
        tarfile.open(fileobj=io.BytesIO(_git(repo, "archive", rev, "src", binary=True))).extractall(d)
        if not (d / "src" / "simplyqrl").exists():          # revisions before the library was vendored: the vendored copy is upstream's
            tarfile.open(fileobj=io.BytesIO(_git(repo, "archive", "d0aa338", "src/simplyqrl", binary=True))).extractall(d)
    if not (d / "experiments").exists():
        exp = _git(repo, "archive", rev, "experiments", binary=True)       # the scripts of the experiments define a few arms (PPO)
        if exp:
            tarfile.open(fileobj=io.BytesIO(exp)).extractall(d)
    return d


def commit_time(repo, rev):
    return int(_git(repo, "log", "-1", "--format=%ct", rev))


def old_labels():
    rows = list(csv.DictReader(open(ROOT / "docs" / "arm_names.csv", encoding="utf-8")))
    def find(config, stage, arm):
        """Old name of the arm in that stage; failing that in that configuration; failing that the only name of the arm (the arms of
        the terminal ablation and of the pilot keep the name of the arm that they repeat)."""
        for ok in (lambda r: config in r["config"].split("|") and stage in r["stage"].split("|"),
                   lambda r: config in r["config"].split("|"),
                   lambda r: True):
            for r in rows:
                if r["new_name"] == arm and ok(r):
                    return r["old_name"]
        return None
    return find


def collect_items(args, phase):
    find = old_labels()
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "runs.csv", encoding="utf-8", newline=""))
            if r["algorithm"] == ("ppo" if phase == "ppo" else "dqn") and r["in_results"] == "true" and r["variant"] == ""]
    revs = {}
    for r in rows:
        arm = r["arm"][:-len("_no_terminal_adjustment")] if r["arm"].endswith("_no_terminal_adjustment") else r["arm"]
        rev = r["effective_revision"] or r["code_revision"] or (EARLY if r["started_utc"] < START_LIBRARY_UTC else FIRST_WITH_LIBRARY)
        revs.setdefault((r["config"], r["stage"], arm), set()).add(rev)
    wanted = sorted(revs) if phase in ("params", "ppo") else [t for t in TRACE_ARMS if t in revs]
    missing = [t for t in TRACE_ARMS if t not in revs] if phase == "traces" else []
    items = []
    for cfg, stage, arm in wanted:
        old = find(cfg, stage, arm)
        order = sorted(revs[(cfg, stage, arm)], key=lambda v: commit_time(args.old_repo, v))
        pick = order if phase in ("params", "ppo") else sorted({order[0], order[-1]}, key=order.index)
        for rev in pick:
            items.append(dict(id=f"{cfg}|{stage}|{arm}|{rev}", config=cfg, stage=stage, arm=arm, old_label=old, rev=rev, seed=1))
    return items, missing


def fill_protocol(items, steps, phase, workdir):
    import tomllib
    for it in items:
        cfg = tomllib.load(open(ROOT / "configs" / f"{it['config']}.toml", "rb"))
        st = next(s for s in cfg["stage"] if s["name"] == it["stage"])
        it["seed"] = int(str(st["seeds"]).replace(",", "-").split("-")[0])           # first seed of the stage
        it["float64"] = it["arm"] != "frozenlake_binary_linear_ceiling"             # this ceiling builds float32 buffers in both codes
        it["env_id"] = {"cartpole": "CartPole-v1", "frozenlake": "FrozenLake4x4Scalar-v0"}[st["environment"]]
        if phase == "ppo":
            it.update(algo="ppo", steps=steps, trace=True, kw={}, terminal=True)
            it["work"] = str(pathlib.Path(workdir) / "scratch" / hashlib.sha1(it["id"].encode()).hexdigest()[:12])
            continue
        if it["arm"] == "frozenlake_onehot_reference":
            it["env_id"] = "FrozenLake4x4OneHot-v0"
        it["terminal"] = bool(st.get("terminal_adjustment", True))
        kw = dict(st.get("dqn", {}))
        if phase == "traces":
            kw["learning_starts"] = min(int(kw.get("learning_starts", 10_000)), 200)
        elif phase == "params":
            kw["learning_starts"] = 10_000
        it["kw"] = kw
        it["steps"] = {"params": 2, "traces": steps, "full": int(st["steps"])}[phase]
        it["trace"] = phase != "params"
        it["work"] = str(pathlib.Path(workdir) / "scratch" / hashlib.sha1(it["id"].encode()).hexdigest()[:12])
    return items


def run_jobs(args, items, tag, jobs):
    base = pathlib.Path(args.workdir)
    (base / "jobs").mkdir(parents=True, exist_ok=True)
    plan = []
    chunk = max(1, int(args.chunk))
    by_old = {}
    for it in items:
        by_old.setdefault(it["rev"], []).append(it)
    for rev, group in by_old.items():
        for i in range(0, len(group), chunk):
            plan.append(("old", rev, group[i:i + chunk]))
    new_items = items
    for i in range(0, len(new_items), chunk):
        plan.append(("new", "", new_items[i:i + chunk]))
    futures = {}
    results = {"old": {}, "new": {}}
    env_base = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def launch(k, side, rev, group):
        f = base / "jobs" / f"{tag}_{k}_{side}.json"
        json.dump(group, open(f, "w"))
        out = base / "jobs" / f"{tag}_{k}_{side}_out.json"
        if side == "old":
            code = export_old(args.old_repo, rev, base)
            env = dict(env_base, PYTHONPATH=os.pathsep.join([str(code / "src"), str(code / "experiments")]))
        else:
            env = dict(env_base, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "third_party")]))
        p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "worker", "--side", side, "--items", str(f), "--out", str(out)],
                           env=dict(env, PYTHONNOUSERSITE="1"), capture_output=True, text=True, cwd=str(base))
        data = json.load(open(out)) if out.exists() else {}
        for it in group:
            if it["id"] not in data:
                data[it["id"]] = {"id": it["id"], "error": "worker did not finish: " + (p.stderr.strip().splitlines() or [""])[-1][:200]}
        return side, data, out

    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(launch, k, *p) for k, p in enumerate(plan)]
        for fu in cf.as_completed(futs):
            side, data, out = fu.result()
            results[side].update({k: dict(v, _out=str(out)) for k, v in data.items()})
            print(f"  {side}: {len(results[side])}/{len(items)}", flush=True)
    return results


def compare(items, res, tol, with_traces):
    import numpy as np
    rows, bad = [], 0
    for it in items:
        o, n = res["old"].get(it["id"], {}), res["new"].get(it["id"], {})
        row = {"id": it["id"], "old_label": o.get("label", it["old_label"]), "rev": it["rev"]}
        if "error" in o or "error" in n:
            row.update(status="ERROR", old=o.get("error"), new=n.get("error")); bad += 1
        else:
            same_p = o["n_params"] == n["n_params"] and [t[1] for t in o["tensors"]] == [t[1] for t in n["tensors"]]
            row.update(n_params_old=o["n_params"], n_params_new=n["n_params"], same_tensors=same_p)
            ok = same_p
            if with_traces:
                a = np.load(pathlib.Path(o["_out"]).with_suffix("").as_posix() + "__" + hashlib.sha1(it["id"].encode()).hexdigest()[:12] + ".npz")
                b = np.load(pathlib.Path(n["_out"]).with_suffix("").as_posix() + "__" + hashlib.sha1(it["id"].encode()).hexdigest()[:12] + ".npz")
                same_a = bool(np.array_equal(a["actions"], b["actions"]))
                wd = max(float(np.abs(a[k] - b[k]).max()) for k in a.files if k.startswith("w")) if ok else None
                first = None if same_a else int(np.argmax(a["actions"][:min(len(a["actions"]), len(b["actions"]))] != b["actions"][:min(len(a["actions"]), len(b["actions"]))]))
                row.update(n_actions=int(len(a["actions"])), weights_moved=o.get("moved"), same_actions=same_a, first_action_difference=first, max_weight_difference=wd)
                ok = ok and same_a and wd is not None and wd <= tol
            row["status"] = "OK" if ok else "DIFFERENT"
            bad += 0 if ok else 1
        rows.append(row)
    return rows, bad


COLUMNS = ["phase", "config", "stage", "arm", "old_label", "revision", "n_params_old", "n_params_new", "same_tensors", "n_actions", "weights_moved",
           "same_actions", "first_action_difference", "max_weight_difference", "status"]


def _export(work, out):
    """One table per phase from the reports of ``work`` (a retry report replaces the items of the report that it repeats)."""
    out.mkdir(parents=True, exist_ok=True)
    groups = {"params": ["report_params.json", "report_params_retry.json"], "traces": ["report_traces_main.json", "report_traces_or1.json"],
              "ppo": ["report_ppo.json"], "full": sorted(p.relative_to(work).as_posix() for p in work.glob("full_*/report_full.json"))}
    for phase, files in groups.items():
        rows = []
        for f in files:
            if (work / f).exists():
                new = json.load(open(work / f))["rows"]
                keys = {r["id"].rsplit("|", 1)[0] for r in new}
                rows = [r for r in rows if r["id"].rsplit("|", 1)[0] not in keys] + new
        with open(out / f"{phase}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore"); w.writeheader()
            for r in sorted(rows, key=lambda r: r["id"]):
                cfg, stage, arm, rev = (r["id"].split("|") + ["", "", "", ""])[:4]
                w.writerow(dict(r, phase=phase, config=cfg, stage=stage, arm=arm, revision=rev))
        print(phase, len(rows), "rows")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)
    w = sub.add_parser("worker"); w.add_argument("--side"); w.add_argument("--items"); w.add_argument("--out")
    for name in ("params", "traces", "full", "ppo"):
        p = sub.add_parser(name)
        p.add_argument("--failed-from", default="", help="repeat only the items that a previous report of this phase did not mark OK")
        p.add_argument("--old-repo", required=True); p.add_argument("--workdir", required=True)
        p.add_argument("--jobs", type=int, default=4); p.add_argument("--chunk", type=int, default=20)
        p.add_argument("--tol", type=float, default=1e-12)
        if name in ("traces", "ppo"):
            p.add_argument("--steps", type=int, default=2000 if name == "traces" else 4096)
        if name == "full":
            p.add_argument("--config", required=True); p.add_argument("--stage", required=True); p.add_argument("--arm", required=True)
        p.add_argument("--only", default="", help="substring of the item id")
    ex = sub.add_parser("export", help="write the reports of a work folder as tables (results/verification/)")
    ex.add_argument("--workdir", required=True); ex.add_argument("--out", default=str(ROOT / "results" / "verification"))
    a = ap.parse_args()
    if a.phase == "worker":
        return _worker(a.side, a.items, a.out)
    if a.phase == "export":
        return _export(pathlib.Path(a.workdir), pathlib.Path(a.out))
    if a.phase == "full":
        find = old_labels()
        rows = [r for r in csv.DictReader(open(ROOT / "results" / "runs.csv", encoding="utf-8", newline=""))
                if (r["config"], r["stage"], r["arm"]) == (a.config, a.stage, a.arm) and r["seed"] == "1"]
        r = rows[0]
        rev = r["effective_revision"] or r["code_revision"] or FIRST_WITH_LIBRARY
        items = [dict(id=f"{a.config}|{a.stage}|{a.arm}|{rev}", config=a.config, stage=a.stage, arm=a.arm, old_label=find(a.config, a.stage, a.arm), rev=rev, seed=1)]
        items = fill_protocol(items, 0, "full", a.workdir)
        for it in items:
            it["kw"] = dict(it["kw"])
        phase = "full"
    else:
        items, missing = collect_items(a, a.phase)
        if missing:
            print("arms of the list that are not in runs.csv:", missing)
        items = fill_protocol(items, getattr(a, "steps", 0), a.phase, a.workdir)
        phase = a.phase
    if a.only:
        items = [i for i in items if a.only in i["id"]]
    suffix = ""
    if getattr(a, "failed_from", ""):
        failed = {r["id"].rsplit("|", 1)[0] for r in json.load(open(a.failed_from))["rows"] if r["status"] != "OK"}     # (configuration, stage, arm)
        items = [i for i in items if i["id"].rsplit("|", 1)[0] in failed]
        suffix = "_retry"
    print(f"{len(items)} items")
    res = run_jobs(a, items, phase + suffix, a.jobs)
    rows, bad = compare(items, res, a.tol, with_traces=phase != "params")
    out = pathlib.Path(a.workdir) / f"report_{phase}{suffix}.json"
    json.dump({"phase": phase, "tolerance": a.tol, "items": len(rows), "not_ok": bad, "rows": rows}, open(out, "w"), indent=1)
    print(f"{len(rows)} items, {bad} not OK -> {out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
