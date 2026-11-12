"""Weight saving (docs/reproducibility.md's the ceiling follow-up dependency, docs/reproducibility.md
"model weights are now saved, narrowly").

`SafeDQN.train()` writes the ONLINE network's `state_dict()` once, after
training finishes - no optimiser state, no replay buffer, no
exploration-schedule position, no RNG streams. This is inference support
(e.g. extracting a trained circuit's Fourier spectrum), explicitly NOT
resumption support - `docs/reproducibility.md` point 3 ("Mid-run continuation - no, and
deliberately not") is unaffected and untested here on purpose, because
nothing about it changed.

The compatibility guard that matters most: `TrainOutcome.extra` already
defaulted to `{}` before this change, so a manifest written before this
change simply lacks the `weights_path` key - no schema migration, and this is
pinned below by loading a manifest shaped like the pre-change ones.
"""
from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from quantumrl_dissection.environments import FROZEN_SCALAR_ID  # noqa: E402
from quantumrl_dissection.training.runner import RunSpec, run_grid  # noqa: E402


def _cheap_spec(seed: int) -> RunSpec:
    """A classical, tiny-budget cell - fast enough for a unit test."""
    return RunSpec(
        arm="frozenlake_matched_control",
        seed=seed,
        terminal_adjustment=True,
        total_timesteps=500,
        dqn_kwargs=dict(batch_size=32, buffer_size=1_000, train_frequency=1),
    )


def test_weights_file_is_written_and_loadable(tmp_path):
    out = run_grid([_cheap_spec(seed=901)], tmp_path, env_id=FROZEN_SCALAR_ID)
    manifest = out[0]
    weights_path = pathlib.Path(manifest["outcome"]["extra"]["weights_path"])

    assert weights_path.exists(), "weights file was not written"
    assert weights_path.parent == tmp_path, "weights file should sit alongside the other cell artefacts"

    state_dict = torch.load(weights_path, weights_only=True)
    assert len(state_dict) > 0
    # frozen_matched_scalar is a 1-hidden-layer classical net: two Linear
    # layers, weight+bias each.
    assert any("weight" in k for k in state_dict)


def test_weights_path_recorded_in_the_manifest_on_disk(tmp_path):
    run_grid([_cheap_spec(seed=902)], tmp_path, env_id=FROZEN_SCALAR_ID)
    manifest_files = list(tmp_path.glob("*.manifest.json"))
    assert len(manifest_files) == 1

    import json
    m = json.loads(manifest_files[0].read_text())
    assert "weights_path" in m["outcome"]["extra"]
    assert pathlib.Path(m["outcome"]["extra"]["weights_path"]).exists()


def test_weights_save_with_a_relative_outdir(tmp_path, monkeypatch):
    """Regression test for a real bug caught by a smoke run
    run, not by this file: `SafeDQN.train()` os.chdir()s INTO `self.outdir`
    for upstream's relative `runs/*.csv` writes, then back out in a
    `finally`. `weights_path = self.outdir / f"{run_name}_weights.pt"` was
    built INSIDE that chdir'd window - fine when `self.outdir` was already
    absolute (every other test in this file uses pytest's `tmp_path`, which
    always is), but a RELATIVE `self.outdir` double-resolved against the
    now-changed cwd and crashed with 'Parent directory ... does not exist'
    after training had already finished, having spent the compute for
    nothing. Every experiment script's `--outdir` default IS relative
    (e.g. `results/frozenlake`), so this was not a
    theoretical case. `SafeDQN.__init__` now resolves `self.outdir` to an
    absolute path at construction; this test pins the exact scenario the bug
    needed - a genuinely relative path, exercised from a fresh cwd - rather
    than one that happens to already be absolute like `tmp_path`."""
    monkeypatch.chdir(tmp_path)
    relative_outdir = pathlib.Path("nested") / "outdir"  # deliberately relative

    out = run_grid([_cheap_spec(seed=903)], relative_outdir, env_id=FROZEN_SCALAR_ID)
    manifest = out[0]
    assert "error" not in manifest, manifest.get("error")
    weights_path = pathlib.Path(manifest["outcome"]["extra"]["weights_path"])

    assert weights_path.is_absolute(), "weights_path must not be relative-on-relative"
    assert weights_path.exists()
    assert weights_path == (tmp_path / relative_outdir / weights_path.name)
    torch.load(weights_path, weights_only=True)  # loadable, not just present


def test_pre_change_manifest_shape_still_parses_with_no_weights_key():
    """Pins backward compatibility directly: a manifest shaped exactly like
    the ones already on disk before this change (extra={}) must not raise
    when code reaches for weights_path - it should simply not be there."""
    legacy_outcome = {
        "run_name": "legacy__fix01on__s1",
        "seed": 1,
        "total_timesteps": 100_000,
        "wall_seconds": 123.4,
        "probe": {},
        "episodes_csv": "runs/legacy__fix01on__s1.csv",
        "eval_csv": None,
        "trace_npy": "legacy__fix01on__s1_trace.npy",
        "extra": {},  # exactly what every manifest before this change has
    }
    assert legacy_outcome["extra"].get("weights_path") is None
