"""Interval estimates, and the bias they exist to avoid.

The demonstration in `test_max_over_training_rewards_noise` is the reason this
module was written: two arms with identical true performance, differing only in
variance, produce an apparent 85% gap under a max-over-training protocol. Our
`best_ma50` and `greedy_best` are exactly that protocol, and the paper's own
numbers show the quantum arms are three times noisier than the classical ones.
"""
import numpy as np
import pytest

from quantumrl_dissection.analysis import stats as S


def test_iqm_ignores_the_tails():
    """One diverging run must not move the estimate. VQ-DQN policies do diverge
    (Franz et al. 2022), so this is the common case, not the pathological one."""
    clean = [100, 105, 110, 115, 120, 125, 130, 135]
    with_outlier = clean[:-1] + [10_000]
    assert abs(S.iqm(clean) - S.iqm(with_outlier)) < 1.0
    assert np.mean(with_outlier) > 3 * np.mean(clean)


def test_iqm_degrades_gracefully_below_four_runs():
    """Trimming 25% from each side of three runs would discard everything."""
    assert S.iqm([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_bootstrap_ci_brackets_the_estimate():
    x = np.random.default_rng(0).normal(200, 30, 10)
    lo, hi = S.bootstrap_ci(x)
    assert lo < S.iqm(x) < hi


def test_ci_is_wider_for_a_noisier_arm():
    """The point of interval estimates: variance shows up in the interval rather
    than being hidden behind a point estimate."""
    rng = np.random.default_rng(0)
    n_lo, n_hi = S.bootstrap_ci(rng.normal(200, 20, 10))
    w_lo, w_hi = S.bootstrap_ci(rng.normal(200, 120, 10))
    assert (w_hi - w_lo) > 2 * (n_hi - n_lo)


def test_probability_of_improvement_is_symmetric_and_calibrated():
    rng = np.random.default_rng(1)
    a, b = rng.normal(200, 50, 40), rng.normal(200, 50, 40)
    p = S.probability_of_improvement(a, b)
    assert 0.35 < p < 0.65
    assert S.probability_of_improvement(a, b) + S.probability_of_improvement(b, a) \
        == pytest.approx(1.0)


def test_probability_of_improvement_detects_a_real_gap():
    rng = np.random.default_rng(2)
    assert S.probability_of_improvement(rng.normal(400, 20, 30),
                                        rng.normal(100, 20, 30)) > 0.95


def test_max_over_training_rewards_noise():
    """Two arms, identical true mean, different variance. The maximum over the
    curve invents an advantage for the noisier one; the mean of a final window
    does not."""
    rng = np.random.default_rng(3)
    noisy = rng.normal(200, 120, (200, 500))
    stable = rng.normal(200, 35, (200, 500))

    assert noisy.max(axis=1).mean() > 1.5 * stable.max(axis=1).mean()

    fp_noisy = np.mean([S.final_performance(r) for r in noisy])
    fp_stable = np.mean([S.final_performance(r) for r in stable])
    assert abs(fp_noisy - fp_stable) < 15, "final-window means must agree"


def test_final_performance_uses_the_end_of_training():
    rising = list(range(100))
    assert S.final_performance(rising, last_frac=0.1) == pytest.approx(94.5)


def test_greedy_final_is_mean_of_last_n_checkpoints():
    checkpoints = [0.0, 0.1, 0.2, 0.9, 0.8, 1.0]  # 6 checkpoints, last 3 = [0.9,0.8,1.0]
    assert S.greedy_final(checkpoints, last_n=3) == pytest.approx((0.9 + 0.8 + 1.0) / 3)


def test_greedy_final_clips_to_available_checkpoints():
    """Fewer checkpoints than last_n: use whatever is there, do not raise or pad."""
    assert S.greedy_final([1.0, 0.5], last_n=3) == pytest.approx(0.75)
    assert S.greedy_final([1.0], last_n=3) == pytest.approx(1.0)
    assert np.isnan(S.greedy_final([], last_n=3))


def test_greedy_final_is_not_a_maximum_unlike_greedy_best():
    """The whole point of the statistic: a single lucky spike late in an
    otherwise-flat checkpoint series should not dominate it the way max()
    would - it must be visibly pulled toward the OTHER recent checkpoints."""
    checkpoints = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # one spike as the last point
    assert max(checkpoints) == 1.0
    gf = S.greedy_final(checkpoints, last_n=3)
    assert gf < 0.5, "one spike among the last 3 must not read like a max"


def test_stratified_bootstrap_keeps_every_task_represented():
    tasks = {"cartpole": list(np.random.default_rng(4).normal(300, 40, 10)),
             "frozenlake": list(np.random.default_rng(5).normal(0.8, 0.1, 10))}
    lo, hi = S.stratified_bootstrap_ci(tasks)
    assert lo < hi
    # and the warning in the docstring is real: the scales are incommensurable
    assert hi > 1.0


def test_summarise_reports_n():
    """An interval without its N is not interpretable: Agarwal et al. validate
    coverage at N=10 and warn that at N=3 the CI is too narrow."""
    out = S.summarise_scores([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert out["n"] == 10
    assert out["ci_low"] < out["iqm"] < out["ci_high"]
