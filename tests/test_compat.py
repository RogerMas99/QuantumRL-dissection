"""Adjustments to the library: the patches are applied and guarded.

Requires torch + simplyqrl to be installed.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("simplyqrl")

import quantumrl_dissection  # noqa: E402  (import applies the patches)
from quantumrl_dissection.arms import ARMS, HYBRID  # noqa: E402

PAPER_LINEAR = ARMS["reference_control"][1]


def test_report_lists_both_patches():
    applied = quantumrl_dissection.upstream_report()["patches_applied"]
    assert "output_scale" in applied
    assert "agent_type_aliases" in applied


def test_classic_alias_resolves():
    """The paper's own experiment script passes agent_type='classic', which
    upstream build_agent cannot resolve (returns None)."""
    from quantumrl_dissection.agents.capacity import build_agent_for
    agent = build_agent_for("classic", PAPER_LINEAR, is_qnet=True)
    assert agent is not None


def test_unknown_agent_type_raises_instead_of_returning_none():
    from simplyqrl.agents import build_agent
    with pytest.raises(ValueError):
        build_agent("does-not-exist", (4,), 2, {}, is_qnet=True)


def test_paper_linear_head_is_actually_linear():
    """3 reused indices x 4 repeats -> Linear(12, 2), no hidden layers."""
    from quantumrl_dissection.agents.capacity import build_agent_for, count_trainable
    agent = build_agent_for("classic", PAPER_LINEAR, is_qnet=True)
    n = count_trainable(agent)
    assert n < 100, f"expected a tiny linear head, got {n} parameters"


def test_output_scaling_actually_reaches_the_model():
    """With use_output_scaling the module must contain an OutputScale."""
    from simplyqrl.agents import OutputScale
    from quantumrl_dissection.agents.capacity import build_agent_for
    cfg = dict(HYBRID)
    cfg["use_output_scaling"] = True
    agent = build_agent_for("hybrid", cfg, is_qnet=True)
    assert any(isinstance(m, OutputScale) for m in agent.modules()), (
        "OutputScale missing: the output-scale adjustment did not take effect"
    )
