"""Interval estimates for small numbers of runs.

Following Agarwal et al., "Deep Reinforcement Learning at the Edge of the
Statistical Precipice" (NeurIPS 2021). Implemented here rather than depending on
`rliable` so the pinned environment stays untouched; the estimators are short and
the point is to get the reporting right, not to add a package.

WHY THIS EXISTS, AND WHAT IT ADDRESSES
----------------------------------
Two problems with mean +/- sd over three seeds, and one with our own metric.

**Mean is dominated by outliers.** A single diverging run moves it a long way.
Deep RL scores are heavy-tailed, and VQ-DQN especially so - Franz et al. (2022)
document policies that diverge outright. IQM (the mean of the middle 50% of runs)
keeps the median's robustness with far less uncertainty.

**Sample standard deviation is not an interval estimate.** Agarwal et al. argue
for percentile bootstrap CIs instead, which are better justified at small N. They
validate good coverage at N = 10, and warn that at N = 3 bootstrap CIs
*underestimate* the true 95% interval. That is the statistical case for plan B,
and it is stronger than "more seeds is nicer".

**And the metric we have been using is biased.** `best_ma50` and `greedy_best`
are MAXIMA over training. A maximum over a noisy curve is positively biased, and
the bias grows with variance - so a noisier arm scores higher for free. The
paper's own numbers show the quantum arms are far noisier than the classical ones
(+/-115 against +/-37 at r32), which means a max-over-training protocol
systematically flatters the quantum side. Agarwal et al. call out exactly this
class of non-standard evaluation protocol as an explanation for apparent gains.

`final_performance` below is the unbiased alternative and needs no retraining:
every episode is already in the CSVs.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "iqm",
    "bootstrap_ci",
    "stratified_bootstrap_ci",
    "probability_of_improvement",
    "final_performance",
    "greedy_final",
    "summarise_scores",
]


def iqm(scores: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50% of runs."""
    x = np.sort(np.asarray(scores, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if x.size < 4:
        return float(np.mean(x))     # trimming below 4 runs discards everything
    lo, hi = int(np.floor(x.size * 0.25)), int(np.ceil(x.size * 0.75))
    return float(np.mean(x[lo:hi]))


def bootstrap_ci(scores: Sequence[float], statistic=iqm, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for one task.

    Agarwal et al. validate good coverage at N = 10 and note that at N = 3 the
    interval is too narrow. Report the N alongside it, always.
    """
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [statistic(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def stratified_bootstrap_ci(scores_by_task: Dict[str, Sequence[float]],
                            statistic=iqm, n_boot: int = 10_000,
                            alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """CI over several tasks, resampling runs independently within each task.

    Stratifying keeps every task represented in each bootstrap sample. Note the
    scale we are working at: Atari 100k gives 26 tasks x 3 runs = 78 samples,
    while CartPole plus FrozenLake gives 2 x 10 = 20. Aggregating across two
    tasks buys much less than the paper's setting, and scores must be
    commensurable first - CartPole returns run 0-500, FrozenLake success 0-1.
    Prefer per-task intervals unless you have normalised deliberately.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(v, dtype=float) for v in scores_by_task.values()]
    arrays = [a[np.isfinite(a)] for a in arrays if len(a)]
    if not arrays:
        return float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        pooled = np.concatenate([rng.choice(a, size=a.size, replace=True) for a in arrays])
        boots.append(statistic(pooled))
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def probability_of_improvement(x: Sequence[float], y: Sequence[float]) -> float:
    """P(a run of X beats a run of Y), ties counted as half.

    The honest way to phrase a comparison at small N. "0.62" says a randomly
    chosen X run beats a randomly chosen Y run 62% of the time - which is a much
    weaker and more accurate claim than "X is better than Y", and it does not
    pretend the two means are separated when the runs overlap.
    """
    a = np.asarray(x, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(y, dtype=float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    wins = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return float(wins / (a.size * b.size))


def final_performance(rewards: Sequence[float], last_frac: float = 0.1) -> float:
    """Mean return over the final `last_frac` of episodes.

    The unbiased counterpart to `best_ma50`. A maximum over a training curve is
    positively biased and the bias grows with variance, so it rewards a noisy arm
    for being noisy. This averages a constant window at the end of training instead,
    which every run reaches on equal terms.
    """
    x = np.asarray(rewards, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(round(x.size * last_frac)))
    return float(np.mean(x[-k:]))


def greedy_final(greedy_checkpoint_scores: Sequence[float], last_n: int = 3) -> float:
    """Mean of the last `last_n` greedy-evaluation checkpoints for one cell.

    Neither statistic already in use is simultaneously free of exploration
    noise AND free of the maximum-over-training bias:

        greedy_best         max() over the greedy-eval checkpoints (epsilon=0,
                             so clean of exploration) - but a MAXIMUM, so
                             positively biased, worse for noisier arms (same
                             argument as `final_performance` vs `best_ma50`).
        final_performance    mean over the LAST TRAINING episodes - not a
                             maximum, but those episodes ran under
                             epsilon-greedy at `end_e` (0.05 by this
                             project's default, never exactly 0), so it is
                             not clean of exploration either.

    `greedy_final` is the mean of the last `last_n` entries of the SAME
    eval-checkpoint series `greedy_best` already uses (`env.step` at
    epsilon=0, `dqn/safe.py::evaluate_greedy`) - clean of exploration like
    `greedy_best`, and a mean rather than a max like `final_performance`.
    Needs only the eval CSV already on disk; no retraining.

    `last_n` is silently clipped to however many checkpoints the cell
    actually has (a short run may have fewer than `last_n`) - callers that
    care whether that happened should check the checkpoint count themselves,
    e.g. `len(load_eval(...)[0])`.
    """
    x = np.asarray(greedy_checkpoint_scores, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = min(last_n, x.size)
    return float(np.mean(x[-k:]))


def summarise_scores(scores: Sequence[float], name: str = "") -> Dict[str, float]:
    """IQM with a 95% percentile bootstrap CI, plus the N that produced it."""
    x = np.asarray(scores, dtype=float)
    x = x[np.isfinite(x)]
    lo, hi = bootstrap_ci(x)
    out = {"n": int(x.size), "iqm": iqm(x), "ci_low": lo, "ci_high": hi,
           "median": float(np.median(x)) if x.size else float("nan"),
           "mean": float(np.mean(x)) if x.size else float("nan")}
    if name:
        out["name"] = name
    return out
