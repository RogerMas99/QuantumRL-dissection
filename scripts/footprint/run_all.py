"""Run the whole calibration schedule sequentially (idle machine, then each kind alone and in parallel).

    python scripts/footprint/run_all.py --outroot calibration_runs --minutes 12
A measurement whose summary.json already exists is skipped, so the schedule can be resumed. Nothing else should run on the
machine while it is going: CodeCarbon measures the whole machine.
"""
import argparse
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
# kind -> processes in the parallel condition (14 workers for the classical and simulator cells, 8 for the real-circuit cells)
PLAN = [("sv_cartpole", 14), ("su2_cartpole", 14), ("classic_dqn", 14), ("ppo", 14), ("real_cartpole", 8), ("real_frozen", 8)]


def run(outroot, kind, cond, procs, minutes):
    out = outroot / f"{kind}__{cond}"
    if (out / "summary.json").exists():
        print("skip", out.name, flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "codecarbon.log", "w") as log:
        subprocess.run([sys.executable, str(REPO / "scripts" / "footprint" / "calibrate.py"), "--kind", kind, "--cond", cond,
                        "--procs", str(procs), "--minutes", str(minutes), "--outroot", str(outroot)], stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))
    print("done", out.name, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outroot", required=True)
    p.add_argument("--minutes", type=float, default=12.0)
    a = p.parse_args()
    outroot = pathlib.Path(a.outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    run(outroot, "idle", "single", 0, 2.0)
    for kind, n in PLAN:
        run(outroot, kind, "single", 1, a.minutes)
        run(outroot, kind, "parallel", n, a.minutes)


if __name__ == "__main__":
    main()
