"""Summarise the calibration and extrapolate to the hours recorded in results/runs.csv.

Inputs : either --calib <dir> with <kind>__<cond>/summary.json and emissions.csv (written by calibrate.py), or
         --calibration-table docs/footprint/calibration_table.csv (the committed table derived from them);
         --runs results/runs.csv (one row per run: wall_seconds, kind, algorithm, environment, arm, log_origin).
Outputs: calibration_table.csv (only with --calib), extrapolation_by_type.csv, extrapolation_totals.csv in --out.

Method (all assumptions are explicit in the columns):
  * kWh per process-hour of a kind = kWh measured by CodeCarbon for the whole machine / (processes x wall-clock hours), per condition
    (single process; N processes in parallel). 'total' includes the GPU (an NVIDIA RTX 5060 Laptop idling at ~9 W: PyTorch is CPU-only,
    so the GPU does no work, but CodeCarbon counts it); 'cpu_ram' = CPU + RAM only.
  * hours of the campaign are the sums of wall_seconds, by kind and by where the run was made (log_origin): the local machine
    ('local') or Colab ('colab'). All runs of runs.csv count, discarded ones included.
  * local machine: condition chosen from the number of runs that shared the machine (concurrency.py): real-circuit runs ran one at a
    time (single-process kWh/h); simulator, classical and PPO runs ran with a mean of 12-18 runs alive (parallel kWh/h, 14 processes).
    'low' = energy above the idle machine (idle row of the calibration), 'high' = energy as measured (machine total, which includes
    the idle baseline of RAM, CPU and the idle GPU while the runs lasted).
  * Colab: hardware unknown. ASSUMPTION: power per process equal to the one measured here for the same kind (range: parallel .. single),
    and a grid intensity range of --colab-ci-low .. --colab-ci-high kg CO2-eq/kWh. These figures are NOT measurements.

    python scripts/footprint/summarize.py --calibration-table docs/footprint/calibration_table.csv --runs results/runs.csv --out <dir>
"""
import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from concurrency import kind_of_run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib")
    p.add_argument("--calibration-table")
    p.add_argument("--runs", default="results/runs.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--es-ci", type=float, default=0.17405, help="kg CO2-eq/kWh applied by CodeCarbon to Spain (global_energy_mix.json, 2023)")
    p.add_argument("--colab-ci-low", type=float, default=0.10)
    p.add_argument("--colab-ci-high", type=float, default=0.45)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if bool(a.calib) == bool(a.calibration_table):
        p.error("give exactly one of --calib and --calibration-table")
    rows = []
    for f in sorted((pathlib.Path(a.calib).glob("*__*/summary.json") if a.calib else [])):
        s = json.loads(f.read_text())
        h = s["codecarbon_duration_s"] / 3600
        n = s["procs"]
        rows.append(dict(kind=s["kind"], cond=s["cond"], procs=n, duration_min=round(s["codecarbon_duration_s"] / 60, 2),
                         kWh_total=s["kWh_total"], kWh_cpu=s["cpu_kWh"], kWh_ram=s["ram_kWh"], kWh_gpu=s["gpu_kWh"],
                         kWh_cpu_ram=s["cpu_kWh"] + s["ram_kWh"], kg_CO2eq=s["kg_CO2eq"],
                         cpu_W_mean=s["cpu_power_W_mean"], ram_W=s["ram_power_W_mean"],
                         kWh_per_proc_hour_total=(s["kWh_total"] / (n * h)) if n else None,
                         kWh_per_proc_hour_cpu_ram=((s["cpu_kWh"] + s["ram_kWh"]) / (n * h)) if n else None,
                         machine_W_mean_total=s["kWh_total"] / h * 1000, steps_total=s["steps_total"],
                         steps_per_s_per_proc=s["steps_per_second_per_process"]))
    if a.calib:
        cal = pd.DataFrame(rows)
        cal.to_csv(out / "calibration_table.csv", index=False)
    else:
        cal = pd.read_csv(a.calibration_table)

    m = pd.read_csv(a.runs)
    m["hours"] = m.wall_seconds / 3600
    m["kind"] = [kind_of_run(r) for r in m.itertuples()]
    m["where"] = m.log_origin.map({"local": "this_PC", "colab": "colab"})
    hrs = m.groupby(["kind", "where"]).hours.sum().reset_index()
    pw = cal[cal.procs > 0].pivot(index="kind", columns="cond", values=["kWh_per_proc_hour_total", "kWh_per_proc_hour_cpu_ram"])
    idle_W = float(cal[cal.kind == "idle"].machine_W_mean_total.iloc[0])
    mw = cal[cal.procs > 0].pivot(index="kind", columns="cond", values="machine_W_mean_total")
    rec = []
    for _, r in hrs.iterrows():
        k, w, H = r["kind"], r["where"], r["hours"]
        if k not in pw.index:
            rec.append(dict(kind=k, where=w, hours=H, note="no calibration for this kind")); continue
        t_par, t_sin = pw.loc[k, ("kWh_per_proc_hour_total", "parallel")], pw.loc[k, ("kWh_per_proc_hour_total", "single")]
        c_par, c_sin = pw.loc[k, ("kWh_per_proc_hour_cpu_ram", "parallel")], pw.loc[k, ("kWh_per_proc_hour_cpu_ram", "single")]
        n_par = int(cal[(cal.kind == k) & (cal.cond == "parallel")].procs.iloc[0])
        if w == "this_PC":
            cond = "single" if k.startswith("real_") else "parallel"
            t, c = (t_sin, c_sin) if cond == "single" else (t_par, c_par)
            n = 1 if cond == "single" else n_par
            inc = max(mw.loc[k, cond] - idle_W, 0) / 1000 / n          # kWh per process-hour above the idle machine
            basis = f"{cond} (measured here; condition from file times)"
            lo_t, hi_t, lo_c, hi_c, lo_inc = t, t, c, c, inc
        else:
            lo_t, hi_t, lo_c, hi_c = min(t_par, t_sin), max(t_par, t_sin), min(c_par, c_sin), max(c_par, c_sin)
            basis = "range parallel..single (Colab: ASSUMPTION)"
            lo_inc = lo_t
        ci_lo, ci_hi = (a.es_ci, a.es_ci) if w == "this_PC" else (a.colab_ci_low, a.colab_ci_high)
        rec.append(dict(kind=k, where=w, hours=H, basis=basis,
                        kWh_low=H * lo_inc, kWh_high=H * hi_t, kWh_cpu_ram=H * (lo_c if w == "this_PC" else hi_c),
                        ci_low=ci_lo, ci_high=ci_hi, kg_low=H * lo_inc * ci_lo, kg_high=H * hi_t * ci_hi,
                        kg_cpu_ram=H * (lo_c if w == "this_PC" else hi_c) * ci_hi))
    ex = pd.DataFrame(rec)
    ex.to_csv(out / "extrapolation_by_type.csv", index=False)
    tot = []
    for w, g in ex.groupby("where"):
        tot.append(dict(where=w, hours=g.hours.sum(), kWh_low=g.kWh_low.sum(), kWh_high=g.kWh_high.sum(), kg_low=g.kg_low.sum(),
                        kg_high=g.kg_high.sum(), kWh_cpu_ram=g.kWh_cpu_ram.sum(), kg_cpu_ram=g.kg_cpu_ram.sum()))
    t = pd.DataFrame(tot)
    t.loc[len(t)] = dict(where="ALL", **{c: t[c].sum() for c in t.columns if c != "where"})
    t.round(3).to_csv(out / "extrapolation_totals.csv", index=False)
    print(cal.round(4).to_string()); print(ex.round(2).to_string()); print(t.round(2).to_string())


if __name__ == "__main__":
    main()
