"""CodeCarbon calibration of one kind of run in one condition (single process / N processes in parallel).

One OfflineEmissionsTracker (country ESP, machine mode, measure_power_secs=5) wraps the whole interval; N worker processes
(scripts/footprint/worker.py, OMP_NUM_THREADS=1) run cells of one kind in a loop; after --minutes they are killed (process tree) and the
tracker is stopped. Outputs, per measurement, in <outroot>/<kind>__<cond>/:
  emissions.csv   CodeCarbon's own file         codecarbon.log   its console log, captured by run_all.py (power-measurement methods are written at start-up)
  summary.json    duration, kWh, kg, kWh per process-hour, steps reached by every worker, steps per second per process
The runs written there are calibration runs (tag 'calib'): they are NOT experiment results and must never be analysed with them.

    python scripts/footprint/calibrate.py --kind sv_cartpole --cond parallel --procs 14 --minutes 12 --outroot calibration_runs
    python scripts/footprint/calibrate.py --kind idle --cond single --procs 0 --minutes 2 --outroot ...      # idle machine
"""
import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]


def steps_reached(folder: pathlib.Path):
    """Sum, over every episode CSV written in <folder>/runs, of the last global_step logged."""
    tot, per = 0, []
    for f in sorted((folder / "runs").glob("*.csv")) if (folder / "runs").exists() else []:
        if f.name.endswith(("_eval.csv", "_diag.csv")):
            continue
        last = 0
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                last = int(float(row.get("global_step") or row.get("step") or last))
        per.append((f.name, last))
        tot += last
    return tot, per


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True)
    p.add_argument("--cond", required=True, choices=["single", "parallel"])
    p.add_argument("--procs", type=int, required=True)
    p.add_argument("--minutes", type=float, default=12.0)
    p.add_argument("--outroot", required=True)
    p.add_argument("--measure-secs", type=int, default=5)
    a = p.parse_args()
    out = pathlib.Path(a.outroot) / f"{a.kind}__{a.cond}"
    out.mkdir(parents=True, exist_ok=True)
    from codecarbon import OfflineEmissionsTracker
    tr = OfflineEmissionsTracker(country_iso_code="ESP", measure_power_secs=a.measure_secs,
                                 output_dir=str(out), output_file="emissions.csv", project_name=f"{a.kind}__{a.cond}",
                                 log_level="info", save_to_file=True, allow_multiple_runs=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    workers = []
    t0 = time.time()
    tr.start()
    for i in range(a.procs):
        wdir = out / f"w{i}"
        cmd = [sys.executable, str(REPO / "scripts" / "footprint" / "worker.py"), "--kind", a.kind, "--seed0", str(9001 + i), "--outdir", str(wdir)]
        workers.append((i, wdir, subprocess.Popen(cmd, env=env, cwd=str(REPO), stdout=open(out / f"w{i}.log", "w"), stderr=subprocess.STDOUT,
                                                  creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))))
    time.sleep(a.minutes * 60)
    for i, wdir, pr in workers:
        subprocess.run(["taskkill", "/PID", str(pr.pid), "/T", "/F"], capture_output=True)
    time.sleep(3)
    kg = tr.stop()
    dur = time.time() - t0
    rows = list(csv.DictReader(open(out / "emissions.csv")))
    last = rows[-1]
    steps, per_worker = 0, []
    for i, wdir, pr in workers:
        s, per = steps_reached(wdir)
        steps += s
        per_worker.append({"worker": i, "steps": s, "runs_touched": len(per)})
    kwh = float(last["energy_consumed"])
    summ = {"kind": a.kind, "cond": a.cond, "procs": a.procs, "wall_seconds_measured_by_script": round(dur, 1),
            "codecarbon_duration_s": float(last["duration"]), "kWh_total": kwh, "kg_CO2eq": float(last["emissions"]),
            "kg_CO2eq_returned_by_stop": kg, "cpu_kWh": float(last["cpu_energy"]), "ram_kWh": float(last["ram_energy"]),
            "gpu_kWh": float(last["gpu_energy"]), "cpu_power_W_mean": float(last["cpu_power"]), "ram_power_W_mean": float(last["ram_power"]),
            "kWh_per_process_hour": (kwh / (a.procs * float(last["duration"]) / 3600.0)) if a.procs else None,
            "steps_total": steps, "steps_per_second_per_process": (steps / (a.procs * float(last["duration"]))) if a.procs else None,
            "per_worker": per_worker, "cpu_model": last.get("cpu_model"), "tracking_mode": last.get("tracking_mode"),
            "country": last.get("country_name"), "codecarbon_version": last.get("codecarbon_version")}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: summ[k] for k in ("kind", "cond", "procs", "codecarbon_duration_s", "kWh_total", "kg_CO2eq", "kWh_per_process_hour", "steps_per_second_per_process")}))


if __name__ == "__main__":
    main()
