"""
The dissection paper's own PPO results, at 10 seeds.

WHERE THESE COME FROM
---------------------
The paper's companion repository (`javier-lazaro/qrl-dissection`) ships the raw
TensorBoard event files for every run behind the published figures: 3 blocks, 36
configurations, 10 seeds each, 360 runs, 100k steps. That is ~140 MB of binary
event files, so we do not vendor them. `scripts/extract_paper_baselines.py`
reduces them to two CSVs in `data/`, and this module reads those.

WHY THIS MATTERS MORE THAN IT SOUNDS
------------------------------------
Until now every "the paper reports X" statement in this repo was a number read
off a figure. Now the comparison is against the authors' own logged returns,
summarised with OUR metric (`best_ma50`, the same rolling window
`core.analysis.summarise_run` applies to our DQN runs). That removes a whole
class of transcription error from the transfer claims, and it supplies the
standard deviations - which turn out to be large enough to matter.

READ THE SPREAD BEFORE READING THE MEANS. Several published contrasts sit inside
one standard deviation of each other at 10 seeds. That is the strongest available
argument for the plan-B robustness commitment in docs/reproducibility.md: if the original
work needs 10 seeds to separate its arms, a 3-seed coverage pass here cannot
settle anything either.

CAVEATS THAT TRAVEL WITH THE DATA
---------------------------------
1. These are PPO. Every comparison against our DQN runs is cross-algorithm by
   construction - that is the point of the thesis, but it is not a like-for-like
   baseline and must never be presented as one.

2. Some configurations are the SAME RUNS under two labels. `Hsiao_OR_r4` and
   `Hsiao_OR_r4_Unentangled` are byte-identical files, as are the r16 pair. That
   is legitimate - Hsiao's default is `ent=False`, so the OR-block runs are the
   unentangled arm of the entanglement block - but it means the two blocks are
   not independent evidence. `reference_study_duplicate_groups()` lists them.

3. `Skolik_DR_L1_Entangled` and `Skolik_DR_L1_Unentangled` hold identical
   returns on all 10 seeds. This is NOT duplicated data: at one layer the `ent`
   flag cannot affect the output at all. See docs/adjustments.md, and
   `tests/test_entanglement_noop.py`, which proves it by construction.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

__all__ = [
    "load_reference_study_baselines",
    "load_reference_study_curves",
    "load_reference_study_curve",
    "load_reference_study_summary",
    "reference_study_config",
    "reference_study_duplicate_groups",
    "compare_to_paper",
    "DATA_DIR",
]

DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "results" / "reference_study"

# Configurations whose logged returns are identical, and why. Kept explicit so a
# reader never mistakes reuse for independent replication.
_DUPLICATES: Dict[str, str] = {
    "Hsiao_OR_r4_Unentangled": "same runs as OR/Quantum_r4 (Hsiao default is ent=False)",
    "Hsiao_OR_r16_Unentangled": "same runs as OR/Quantum_r16 (Hsiao default is ent=False)",
    "Skolik_DR_L1_Entangled": "same runs as DR/Skolik_4Q_L1 (Skolik default is ent=True)",
    "Skolik_DR_L2_Entangled": "same runs as DR/Skolik_4Q_L2 (Skolik default is ent=True)",
    "Skolik_DR_L1_Unentangled": (
        "identical RETURNS to the entangled arm, but a separate run: at "
        "n_layers_q=1 the ent flag is a no-op (the final-ring property)"
    ),
}


def load_reference_study_baselines(path: Optional[pathlib.Path] = None):
    """Per-seed rows: block, config, seed, best_ma50, final_ma50, mean_return."""
    import pandas as pd

    return pd.read_csv(path or DATA_DIR / "reference_study_ppo_baselines.csv")


def load_reference_study_summary(path: Optional[pathlib.Path] = None):
    """Per-configuration aggregate over 10 seeds, with standard deviations."""
    import pandas as pd

    return pd.read_csv(path or DATA_DIR / "reference_study_ppo_summary.csv")


def load_reference_study_curves(per_seed: bool = False, path: Optional[pathlib.Path] = None):
    """Learning curves at 500-step resolution.

    `per_seed=False` (default) returns the per-configuration mean and standard
    deviation across the 10 seeds - what a figure with a shaded band needs.
    `per_seed=True` returns every seed separately, for spaghetti plots or a
    different aggregation.
    """
    import pandas as pd

    name = "reference_study_ppo_curves.csv.gz" if per_seed else "reference_study_ppo_curves_summary.csv.gz"
    return pd.read_csv(path or DATA_DIR / name)


def load_reference_study_curve(config: str, per_seed: bool = False):
    """One configuration's curve, ready to plot against one of ours."""
    df = load_reference_study_curves(per_seed=per_seed)
    hit = df[df.config == config]
    if hit.empty:
        raise KeyError(f"unknown paper config {config!r}. "
                       f"Known: {sorted(df.config.unique())}")
    return hit.sort_values("step")


def reference_study_config(name: str) -> Dict[str, float]:
    """One configuration's summary row as a plain dict.

    >>> reference_study_config("Skolik_8Q_L5")["best_ma50_mean"]
    """
    s = load_reference_study_summary()
    hit = s[s.config == name]
    if hit.empty:
        raise KeyError(f"unknown paper config {name!r}. Known: {sorted(s.config)}")
    return hit.iloc[0].to_dict()


def reference_study_duplicate_groups() -> Dict[str, str]:
    """Configurations that are not independent runs, and the reason for each."""
    return dict(_DUPLICATES)


def compare_to_paper(our_best: float, config_name: str,
                     our_sd: Optional[float] = None) -> str:
    """One line of honest comparison against a published configuration.

    Deliberately reports the paper's spread and refuses to call a difference a
    difference when it sits inside the combined spread. The whole repo exists
    because unguarded comparisons are how the original conclusions got their
    confidence.
    """
    p = reference_study_config(config_name)
    mean, sd, n = p["best_ma50_mean"], p["best_ma50_sd"], int(p["n_seeds"])
    delta = our_best - mean
    pooled = sd if our_sd is None else ((sd ** 2 + our_sd ** 2) / 2) ** 0.5
    verdict = "within spread" if abs(delta) < pooled else "outside spread"
    return (f"{config_name}: paper (PPO, n={n}) {mean:.1f} +/- {sd:.1f} | "
            f"ours (DQN) {our_best:.1f} | delta {delta:+.1f} - {verdict}")


def reference_study_results_for_arm(arm: str) -> Dict[str, float]:
    """Summary row for the paper configuration a `paper_*` arm replicates.

    The bridge that makes a like-for-like comparison possible without anyone
    having to remember that `porting_skolik_8q_depth5` corresponds to the directory
    `DR/Skolik_8Q_L5`.
    """
    from quantumrl_dissection.arms import REFERENCE_STUDY_RESULTS

    if arm not in REFERENCE_STUDY_RESULTS:
        raise KeyError(
            f"{arm!r} does not replicate a paper configuration. "
            f"Arms that do: {sorted(REFERENCE_STUDY_RESULTS)}"
        )
    return reference_study_config(REFERENCE_STUDY_RESULTS[arm])


def reference_study_depth_curve(template: str = "Skolik_8Q") -> List[tuple]:
    """The paper's DR-depth sweep for one template: [(depth, mean, sd), ...].

    The direct comparator for the depth sweep. Note the confound recorded in
    the final-ring property: on the Skolik template, depth L carries only L-1
    effective entangling blocks, so a DR sweep also sweeps entanglement.
    """
    s = load_reference_study_summary()
    rows = s[s.config.str.startswith(template + "_L")]
    out = []
    for _, r in rows.iterrows():
        out.append((int(r.config.rsplit("_L", 1)[1]),
                    float(r.best_ma50_mean), float(r.best_ma50_sd)))
    return sorted(out)
