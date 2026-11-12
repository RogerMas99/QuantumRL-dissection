"""Calibration worker: runs cells of ONE kind in a loop (consecutive seeds) until the orchestrator kills it.

Each kind is a stage of a configuration of ``configs/`` and one arm of it, with the protocol of that stage (steps, evaluation,
DQN hyperparameters); only the stopping rule differs. The diagnostics of the learning-rate experiment are switched off.

  real_cartpole  hybrid (real PennyLane circuit, 8 qubits, depth 5, entangled)        CartPole    rq2_capacity / capacity_ladder
  real_frozen    frozenlake_binary_depth5 (real circuit, 4 qubits, depth 5)           FrozenLake  rq5_frozenlake / frozenlake_arms
  sv_cartpole    sim_statevector_hybrid (batched state-vector simulator)              CartPole    lr_experiment / dqn_simulator_hybrid
  su2_cartpole   sim_emulator_hybrid_unentangled (SU(2) emulator, no entanglement)    CartPole    additional_x100 / x100_cartpole_1
  classic_dqn    classical_twin_reuse8 (classical twin with output reuse 8)           CartPole    lr_experiment / dqn_twin
  ppo            ppo_reference_reuse4_clip (reference PPO control)                    CartPole    lr_experiment / ppo

    python scripts/footprint/worker.py --kind sv_cartpole --seed0 9001 --outdir <dir>

Outputs go to <outdir>, never to a results directory; the runs carry the tag 'calib'.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "third_party"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quantumrl_dissection.training.experiment import load_config, run_stage, stage_cells  # noqa: E402

KINDS = {  # kind: (configuration, stage, arm)
    "real_cartpole": ("rq2_capacity", "capacity_ladder", "hybrid"),
    "real_frozen": ("rq5_frozenlake", "frozenlake_arms", "frozenlake_binary_depth5"),
    "sv_cartpole": ("lr_experiment", "dqn_simulator_hybrid", "sim_statevector_hybrid"),
    "su2_cartpole": ("additional_x100", "x100_cartpole_1", "sim_emulator_hybrid_unentangled"),
    "classic_dqn": ("lr_experiment", "dqn_twin", "classical_twin_reuse8"),
    "ppo": ("lr_experiment", "ppo", "ppo_reference_reuse4_clip"),
}


def stage_of(kind: str):
    config, stage_name, arm = KINDS[kind]
    cfg = load_config(ROOT / "configs" / f"{config}.toml")
    stage = next(s for s in cfg["stage"] if s["name"] == stage_name)
    stage = dict(stage, diagnostics=False, tag="calib", arms=[arm])
    return stage, arm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=list(KINDS))
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override the number of steps of the stage")
    a = ap.parse_args()
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stage, arm = stage_of(a.kind)
    if a.steps:
        stage["steps"] = a.steps
    seed = a.seed0
    while True:
        cells = stage_cells(dict(stage, seeds=[seed]))
        run_stage(stage, cells, out)
        seed += 1000


if __name__ == "__main__":
    main()
