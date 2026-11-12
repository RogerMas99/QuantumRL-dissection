"""the matching recipe: parameter-budget matching."""
import pytest

from conftest import load_by_path

match_hidden_width = load_by_path(
    "_capacity", "src/quantumrl_dissection/agents/capacity.py").match_hidden_width


def test_matched_width_spends_close_to_budget():
    width, spent = match_hidden_width(target_params=120, in_dim=3, out_dim=2)
    assert width >= 1
    assert abs(spent - 120) <= max(3, 0.1 * 120)


def test_width_grows_with_budget():
    small, _ = match_hidden_width(120, in_dim=3, out_dim=2)
    large, _ = match_hidden_width(1200, in_dim=3, out_dim=2)
    assert large > small


def test_rejects_nonsense_budget():
    with pytest.raises(ValueError):
        match_hidden_width(0, in_dim=3, out_dim=2)


def test_total_match_is_close_to_hybrid_total():
    """matched_classical (match_to='total') should land near the hybrid's full
    parameter count (126 for the Fig. 4 config: 80 quantum + 46 head), not just
    the quantum part. Pure arithmetic check, no torch needed."""
    width, spent = match_hidden_width(target_params=126, in_dim=12, out_dim=2)
    assert abs(spent - 126) <= 10, f"expected ~126 total params, got {spent}"
    assert width == 8


def test_matched_arm_in_dim_accounts_for_output_reuse():
    """Regression guard: the matched arm's first Linear sees
    len(indices) * n_repeats features (SelectiveOutputReuse expands the input),
    so sizing must use that in_dim. Using len(indices) alone made the arm ~2.5x
    too big. Here: indices=[1,2,3], n_repeats=4 -> in_dim=12, target 126 -> the
    REAL built network must land near 126, not ~317."""
    in_dim = 3 * 4  # len(indices) * n_repeats
    width, _ = match_hidden_width(126, in_dim=in_dim, out_dim=2)
    real_params = in_dim * width + width + width * 2 + 2  # Linear(in,w)+Linear(w,2)
    assert abs(real_params - 126) <= 10, (
        f"matched arm built {real_params} params, expected ~126 - "
        "check in_dim includes n_repeats"
    )
    assert width == 8



def test_fair_matched_control_uses_full_observation():
    """The fair control must see all 4 observations (in_dim = 4 * n_repeats = 16),
    not the paper's amputated 3. Cart position is a termination variable; hiding
    it would make the arm fail from blindness rather than from being classical.
    Param count must still land near the hybrid's 126."""
    width, _ = match_hidden_width(126, in_dim=16, out_dim=2)   # full obs
    real = 16 * width + width + width * 2 + 2
    assert abs(real - 126) <= 15, f"fair control built {real} params, expected ~126"
    assert width == 7


def test_or_and_dr_configs_are_wellformed():
    """OR and DR sweeps must produce valid hybrid configs that differ from the
    base only in the intended knob. Pure dict check, no torch."""
    from quantumrl_dissection.arms import (
        HYBRID, hybrid_reuse_config, hybrid_depth_config, REUSE_FACTORS as OR_REPEATS, DEPTHS as DR_DEPTHS,
    )
    for R in OR_REPEATS:
        cfg = hybrid_reuse_config(R)
        assert cfg["reuse_repetitions"] == R
        assert cfg["n_layers_q"] == HYBRID["n_layers_q"]  # only OR knob moved
    for L in DR_DEPTHS:
        cfg = hybrid_depth_config(L)
        assert cfg["n_layers_q"] == L
        assert "reuse_repetitions" not in cfg  # only DR knob moved
