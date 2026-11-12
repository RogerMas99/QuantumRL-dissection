import pathlib
"""A finished cell may only be reused if it answers the same question.

This exists because it did not hold. `run_name` encodes arm, the terminal adjustment state, seed
and tag - readable, and an incomplete key. A 1,500-step smoke cell and a
100,000-step production cell get the same filename, so the old
`if manifest.exists(): skip` reported the smoke numbers as the production run,
with different batch size and buffer size, and printed nothing.

That is the worst class of bug this repository can have: not a crash, but a
plausible number attached to the wrong experiment. Hence a guard, and hence a
test for the guard.
"""
import json

import pytest

from quantumrl_dissection.training.runner import REUSE_KEYS, reuse_or_none


def _write(tmp_path, name, spec, env_id="CartPole-v1"):
    p = tmp_path / f"{name}.manifest.json"
    p.write_text(json.dumps({"spec": spec, "run_name": name, "env_id": env_id,
                             "outcome": {}}))
    return p


BASE = {"arm": "hybrid", "seed": 1, "terminal_adjustment": True,
        "total_timesteps": 100_000, "dqn_kwargs": {"batch_size": 128}, "tag": ""}


def test_missing_manifest_is_not_a_hit(tmp_path):
    assert reuse_or_none(tmp_path / "nope.manifest.json", BASE) is None


def test_identical_spec_is_reused(tmp_path):
    p = _write(tmp_path, "run", BASE)
    assert reuse_or_none(p, dict(BASE)) is not None


def test_shorter_run_does_not_satisfy_a_longer_request(tmp_path):
    """The original bug, verbatim."""
    p = _write(tmp_path, "run", dict(BASE, total_timesteps=1_500))
    with pytest.raises(RuntimeError, match="total_timesteps"):
        reuse_or_none(p, BASE)


def test_longer_run_does_not_silently_satisfy_a_shorter_one_either(tmp_path):
    """Deliberately symmetric. A 100k run is not a 60k run: every metric here is
    computed over the whole trace, so the extra steps change the answer."""
    p = _write(tmp_path, "run", dict(BASE, total_timesteps=1_000_000))
    with pytest.raises(RuntimeError, match="total_timesteps"):
        reuse_or_none(p, BASE)


@pytest.mark.parametrize("field,value", [
    ("arm", "large_network"),
    ("seed", 7),
    ("terminal_adjustment", False),
    ("dqn_kwargs", {"batch_size": 32}),
    ("tag", "v2"),
])
def test_every_material_field_is_compared(field, value, tmp_path):
    p = _write(tmp_path, "run", dict(BASE, **{field: value}))
    with pytest.raises(RuntimeError, match=field):
        reuse_or_none(p, BASE)


def test_environment_mismatch_is_caught(tmp_path):
    """the FrozenLake experiment runs two arms on two different registered ids in one grid."""
    p = _write(tmp_path, "run", BASE, env_id="FrozenLake4x4OneHot-v0")
    with pytest.raises(RuntimeError, match="env_id"):
        reuse_or_none(p, BASE, env_id="FrozenLake4x4Scalar-v0")


def test_rerun_mode_returns_none_instead_of_raising(tmp_path):
    p = _write(tmp_path, "run", dict(BASE, total_timesteps=1_500))
    assert reuse_or_none(p, BASE, on_mismatch="rerun") is None


def test_skip_mode_reuses_anyway(tmp_path):
    """Escape hatch for reading old results deliberately. Never a default."""
    p = _write(tmp_path, "run", dict(BASE, total_timesteps=1_500))
    assert reuse_or_none(p, BASE, on_mismatch="skip") is not None


def test_absent_keys_are_not_compared(tmp_path):
    """Scripts that roll their own run_one pass a partial spec; they should be
    able to check the fields they know without being punished for the rest."""
    p = _write(tmp_path, "run", BASE)
    assert reuse_or_none(p, {"seed": 1, "total_timesteps": 100_000}) is not None


def test_reuse_keys_cover_what_defines_a_run():
    for key in ("arm", "seed", "terminal_adjustment", "total_timesteps", "dqn_kwargs", "tag"):
        assert key in REUSE_KEYS


# ---------------------------------------------------------------------------
# Legacy manifests. Early scripts wrote {name, seed, terminal_adjustment, outcome,
# config} and nothing else. Absence of a field is not evidence of a mismatch,
# so refusing outright would recompute hours of finished simulation - but
# accepting silently could report one run's numbers as another's. Hence a third outcome.
# ---------------------------------------------------------------------------

LEGACY = {"name": "hybrid_depth5__s1", "seed": 1, "terminal_adjustment": True,
          "outcome": {"total_timesteps": 100_000}, "config": {}}


def _write_raw(tmp_path, name, payload):
    p = tmp_path / f"{name}.manifest.json"
    p.write_text(json.dumps(payload))
    return p


def test_legacy_manifest_raises_a_different_error(tmp_path):
    """It must not be confused with a real mismatch: the remedy differs."""
    p = _write_raw(tmp_path, "run", LEGACY)
    with pytest.raises(RuntimeError, match="does not record"):
        reuse_or_none(p, {"seed": 1, "dqn_kwargs": {"batch_size": 128}})


def test_legacy_mode_accepts_it(tmp_path):
    p = _write_raw(tmp_path, "run", LEGACY)
    assert reuse_or_none(p, {"seed": 1, "dqn_kwargs": {"batch_size": 128}},
                         on_mismatch="legacy") is not None


def test_step_budget_is_recovered_from_the_outcome(tmp_path):
    """The one field that IS recoverable from a legacy manifest, so a genuine
    step-budget mismatch is still caught rather than excused as legacy."""
    p = _write_raw(tmp_path, "run", LEGACY)
    assert reuse_or_none(p, {"seed": 1, "total_timesteps": 100_000}) is not None
    with pytest.raises(RuntimeError, match="total_timesteps"):
        reuse_or_none(p, {"seed": 1, "total_timesteps": 60_000})


def test_a_real_mismatch_beats_a_legacy_gap(tmp_path):
    """If one field contradicts and another is merely absent, report the
    contradiction: it is the one that would corrupt a result."""
    p = _write_raw(tmp_path, "run", LEGACY)
    with pytest.raises(RuntimeError, match="total_timesteps"):
        reuse_or_none(p, {"seed": 1, "total_timesteps": 60_000,
                          "dqn_kwargs": {"batch_size": 128}})


# ---------------------------------------------------------------------------
# Cooperative claiming across parallel sessions.
# ---------------------------------------------------------------------------

def test_claim_is_exclusive_then_released(tmp_path):
    from quantumrl_dissection.training.runner import claim_cell, release_cell
    mp = tmp_path / "run.manifest.json"
    assert claim_cell(mp, owner="A") is True
    assert claim_cell(mp, owner="B") is False, "two sessions must not hold one cell"
    release_cell(mp)
    assert claim_cell(mp, owner="B") is True, "released cell must be claimable"


def test_stale_claims_are_reclaimed(tmp_path):
    """A session that dies must not reserve a cell forever - with multi-hour
    cells and flaky Colab runtimes, that would strand work indefinitely."""
    import json as _json
    import time as _time
    from quantumrl_dissection.training.runner import _lock_path, claim_cell

    mp = tmp_path / "run.manifest.json"
    _lock_path(mp).write_text(_json.dumps({"owner": "dead", "t": _time.time() - 3600 * 30}))
    assert claim_cell(mp, ttl_hours=12, owner="B") is True


def test_fresh_claims_are_respected(tmp_path):
    import json as _json
    import time as _time
    from quantumrl_dissection.training.runner import _lock_path, claim_cell

    mp = tmp_path / "run.manifest.json"
    _lock_path(mp).write_text(_json.dumps({"owner": "alive", "t": _time.time()}))
    assert claim_cell(mp, ttl_hours=12, owner="B") is False


def test_unreadable_lock_is_treated_as_stale(tmp_path):
    """Drive can leave a partial file. Refusing forever on garbage would be worse
    than the duplicated compute of claiming it."""
    from quantumrl_dissection.training.runner import _lock_path, claim_cell

    mp = tmp_path / "run.manifest.json"
    _lock_path(mp).write_text("{not json")
    assert claim_cell(mp, owner="B") is True


def test_run_stage_passes_claim_to_run_grid_and_not_to_the_evaluation_configuration():
    """--claim must reach ``run_grid`` (which locks cells), not ``GreedyEvalConfig`` (which would raise)."""
    import inspect
    from quantumrl_dissection.training import experiment
    src = inspect.getsource(experiment.run_stage)
    assert "run_grid(specs, outdir, env_id=env_id, eval_cfg=eval_cfg, progress_bar=progress_bar, claim=claim)" in src
    assert "GreedyEvalConfig(" in src and "claim" not in src.split("GreedyEvalConfig(")[1].split(")")[0]


def test_every_configuration_is_resumable_by_the_reuse_guard():
    """Each cell of every configuration has a run name that the guard can compare, and names are unique within a stage."""
    import pathlib
    from quantumrl_dissection.training.experiment import config_cells, load_config
    for path in sorted((pathlib.Path(__file__).resolve().parents[1] / "configs").glob("*.toml")):
        cfg = load_config(path)
        for stage, cells in config_cells(cfg):
            names = [spec.run_name for _, spec in cells]
            assert len(names) == len(set(names)), (path.name, stage["name"])
