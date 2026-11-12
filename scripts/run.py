"""Run an experiment configuration.

    python scripts/run.py --config configs/rq3_output_reuse.toml --outdir <raw runs> --seeds 1-3
    python scripts/run.py --config configs/rq3_output_reuse.toml --list          # print the cells and exit
    python scripts/run.py --config configs/lr_experiment.toml --outdir <raw> --stage lr_experiment_2 --shard 0 8

Runs are resumable: a cell whose manifest exists for the same specification is skipped. Raw artefacts go to ``--outdir``.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "third_party"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quantumrl_dissection.training.experiment import config_cells, load_config, parse_seeds, run_stage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", help="directory for the raw artefacts of the runs")
    ap.add_argument("--stage", default=None, help="run only this stage")
    ap.add_argument("--seeds", default=None, help='restrict the seeds, e.g. "1-3"')
    ap.add_argument("--arms", nargs="+", default=None, help="restrict the arms")
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None, help="run cells I, I+N, ... of N shards")
    ap.add_argument("--claim", action="store_true", help="cooperative locking when several workers share the output directory")
    ap.add_argument("--list", action="store_true", help="print the run names and exit")
    a = ap.parse_args()

    cfg = load_config(a.config)
    seeds = parse_seeds(a.seeds) if a.seeds else None
    total = 0
    for stage, cells in config_cells(cfg, a.stage, seeds, a.arms):
        if a.list:
            for _, spec in cells:
                print(f"{stage['name']}\t{spec.run_name}")
            total += len(cells)
            continue
        if not a.outdir:
            ap.error("--outdir is required to run")
        print(f"== {stage['name']}: {len(cells)} cells -> {a.outdir}", flush=True)
        run_stage(stage, cells, pathlib.Path(a.outdir), shard=tuple(a.shard) if a.shard else None, claim=a.claim)
    if a.list:
        print(f"{total} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
