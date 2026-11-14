"""Run the cells of an experiment of experiments.toml.

    python experiments/run.py --list
    python experiments/run.py --experiment rq2_capacity --outdir runs --seeds 1-3
    python experiments/run.py --experiment lr_experiment --stage dqn_twin --outdir runs --shard 0 8

The episode and greedy-evaluation CSVs of each cell go to ``<outdir>/runs``; its metrics are appended to ``<outdir>/runs.csv``.
A cell that already has a row there is skipped, so an interrupted experiment can be restarted.
"""
import argparse
import pathlib
import tomllib

import pandas as pd

from quantumrl_dissection.analysis import run_metrics
from quantumrl_dissection.arms import ENVIRONMENTS, environment_of
from quantumrl_dissection.learning_rate import PPOSpec, run_ppo_cell
from quantumrl_dissection.training import GreedyEvalConfig, RunSpec, run_cell

CONFIG = pathlib.Path(__file__).with_name("experiments.toml")
COLUMNS = ["experiment", "stage", "arm", "seed", "training", "greedy", "best_ma50", "auc", "solved", "hours"]


def parse_seeds(text):
    """"1-3,7" -> [1, 2, 3, 7]."""
    out = []
    for part in str(text).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def load_experiments():
    with open(CONFIG, "rb") as fh:
        return {e["name"]: e for e in tomllib.load(fh)["experiment"]}


def stage_cells(stage, seeds=None, arms=None):
    """The cells of a stage as (environment id, spec), ordered by seed and arm."""
    cells = []
    for seed in (s for s in parse_seeds(stage["seeds"]) if seeds is None or s in seeds):
        for arm in (a for a in stage["arms"] if arms is None or a in arms):
            if stage["algorithm"] == "ppo":
                cells.append((ENVIRONMENTS["cartpole"]["env_id"], PPOSpec(arm=arm, seed=seed, total_timesteps=int(stage["steps"]))))
                continue
            overrides = {"lr_seed_sampler": True} if stage.get("seed_sampler") else {}
            overrides.update(stage.get("arm_overrides", {}).get(arm, {}))
            cells.append((environment_of(arm), RunSpec(
                arm=arm, seed=seed, terminal_adjustment=bool(stage.get("terminal_adjustment", True)), total_timesteps=int(stage["steps"]),
                dqn_kwargs=dict(stage.get("dqn", {})), tag=stage.get("tag", ""), overrides=overrides)))
    return cells


def run_cells(experiment, stage, cells, outdir):
    table = outdir / "runs.csv"
    done = pd.read_csv(table) if table.exists() else pd.DataFrame(columns=COLUMNS)
    for k, (env_id, spec) in enumerate(cells, 1):
        key = (experiment, stage["name"], spec.arm, spec.seed)
        if ((done.experiment == key[0]) & (done.stage == key[1]) & (done.arm == key[2]) & (done.seed == key[3])).any():
            continue
        print(f"[{k}/{len(cells)}] {spec.run_name}", flush=True)
        eval_episodes = int(stage.get("eval_episodes", 5))
        if stage["algorithm"] == "ppo":
            wall = run_ppo_cell(spec, outdir, eval_episodes=eval_episodes)["wall_seconds"]
        else:
            cfg = GreedyEvalConfig(env_id=env_id, every_steps=int(stage.get("eval_every", 10_000)), n_episodes=eval_episodes)
            wall = run_cell(spec, outdir, env_id=env_id, eval_cfg=cfg).wall_seconds
        runs = outdir / "runs"
        evals = runs / f"{spec.run_name}_eval.csv"
        m = run_metrics(pd.read_csv(runs / f"{spec.run_name}.csv"), pd.read_csv(evals) if evals.exists() else None)
        if "frozenlake" not in spec.arm:
            m["solved"] = None
        row = dict(zip(COLUMNS[:4], key), **m, hours=wall / 3600.0)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(table, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", help="name of the experiment")
    ap.add_argument("--stage", help="run only this stage")
    ap.add_argument("--seeds", help='restrict the seeds, e.g. "1-3"')
    ap.add_argument("--arms", nargs="+", help="restrict the arms")
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), help="run cells I, I+N, ... of N shards")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--list", action="store_true", help="print the stages and the number of cells, and exit")
    a = ap.parse_args()

    experiments = load_experiments()
    names = [a.experiment] if a.experiment else list(experiments)
    for name in names:
        for stage in experiments[name]["stage"]:
            if a.stage and stage["name"] != a.stage:
                continue
            cells = stage_cells(stage, parse_seeds(a.seeds) if a.seeds else None, a.arms)
            if a.shard:
                cells = [c for k, c in enumerate(cells) if k % a.shard[1] == a.shard[0]]
            if a.list:
                print(f"{name}/{stage['name']}: {len(cells)} cells")
            elif a.experiment:
                run_cells(name, stage, cells, pathlib.Path(a.outdir).resolve())


if __name__ == "__main__":
    main()
