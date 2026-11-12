"""PPO of the reference study with per-group lr, optional / modified clipping (training/ppo.py).

What must not silently drift: that LRPPO with its defaults IS the vendored PPO (same episodes, bit for bit); that the
classical control built through the observation wrapper is the reference study's control (same initial weights, same
function); that the collapsed agent is the reuse-R agent with V0 = sum_r W_r0 and leaves torch's RNG stream where it was;
that without clipping and with the modified clipping rule, reuse R and lr x R (collapsed head) follow the same path
in float64; that the usual rule does NOT (so the modification is needed); and that the clipping diagnostic is right.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

import quantumrl_dissection  # noqa: E402,F401
from quantumrl_dissection.training import ppo as LP  # noqa: E402
from simplyqrl.agents import ClassicalAgent  # noqa: E402
from simplyqrl.ppo import PPO  # noqa: E402

SMALL = dict(LP.REFERENCE_PPO_CONFIG, num_steps=128, num_minibatches=4)       # quick, same algorithm


@pytest.fixture(autouse=True)
def _dtype_and_cwd(monkeypatch, tmp_path):
    d = torch.get_default_dtype()
    monkeypatch.chdir(tmp_path)
    yield
    torch.set_default_dtype(d)


def _ppo(arm, seed=1, steps=0, f64=False, name=None, **kw):
    if f64:
        torch.set_default_dtype(torch.float64)
    a = LP.PPO_ARMS[arm]
    ppo = LP.LRPPO(env=LP.make_selected_env(), seed=seed, agent_type="mlp",
                   agent_config={"n_repeats": a.R, "net_arch": []}, run_name=name or f"{arm}_{seed}",
                   head_lr_mult=float(a.R) if a.collapse else 1.0, collapse=a.collapse, clip_mode=a.clip_mode,
                   clip_R=a.R, float64=f64, **{**SMALL, **kw})
    if steps:
        ppo.train(total_timesteps=steps)
    return ppo


def _effective(ppo):
    """Flat vector: V_eff and bias of the actor and the critic."""
    out = []
    for seq in (ppo.agent.actor, ppo.agent.critic):
        w, b = seq[0].weight.detach().double(), seq[0].bias.detach().double()
        R = LP.reuse_of(ppo.agent)
        o, i = w.shape
        out += [(w.reshape(o, R, i // R).sum(dim=1) if R > 1 else w).reshape(-1), b.reshape(-1)]
    return torch.cat(out).numpy()


# ------------------------------------------------------------------------------------------------ the arms table
def test_arms_table():
    names = list(LP.PPO_ARMS)
    assert len(names) == 5 + 5 + 4 + 4 + 2 == 20
    import pathlib
    from quantumrl_dissection.training.experiment import config_cells, load_config
    cfg = load_config(pathlib.Path(__file__).resolve().parents[1] / "configs" / "lr_experiment.toml")
    stage, cells = next((st, c) for st, c in config_cells(cfg, stage="ppo"))
    assert len(cells) == 600 and {sp.arm for _, sp in cells} == set(LP.PPO_ARMS)
    assert LP.PPO_ARMS["ppo_reference_lr_x32_modified_clip"].clip_mode == "modified" and LP.PPO_ARMS["ppo_reference_lr_x32_modified_clip"].collapse
    assert not LP.PPO_ARMS["ppo_reference_reuse8_clip"].collapse and LP.PPO_ARMS["ppo_reference_reuse8_clip"].clip_mode == "torch"
    assert LP.PPO_ARMS["ppo_reference_reuse8_noclip"].clip_mode == "none"
    assert LP.REFERENCE_PPO_CONFIG["num_minibatches"] == 16 and LP.REFERENCE_PPO_CONFIG["max_grad_norm"] == 0.5


# -------------------------------------------------- the control is the reference study's control (wrapper == upstream)
@pytest.mark.parametrize("R", [4, 16])
def test_wrapper_agent_equals_reference_agent(R):
    torch.manual_seed(7)
    ref = PPO(env_id="CartPole-v1", seed=7, agent_type="mlp",
              agent_config={"reuse_indices": [1, 2, 3], "n_repeats": R, "net_arch": []}, run_name="ref", **SMALL)
    mine = LP.LRPPO(env=LP.make_selected_env(), seed=7, agent_type="mlp",
                    agent_config={"n_repeats": R, "net_arch": []}, run_name="mine", **SMALL)
    for (n1, p1), (n2, p2) in zip(ref.agent.named_parameters(), mine.agent.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    x = torch.randn(32, 4)
    with torch.no_grad():
        assert torch.equal(ref.agent.actor(ref.agent.input_reuse(x)), mine.agent.actor(mine.agent.input_reuse(x[:, 1:])))
        assert torch.equal(ref.agent.get_value(x), mine.agent.get_value(x[:, 1:]))
    assert sum(p.numel() for p in mine.agent.parameters()) == (3 * R + 1) * 2 + (3 * R + 1)   # actor 2 rows + critic 1


def test_defaults_reproduce_vendored_ppo_bit_for_bit():
    kw = dict(SMALL, num_steps=256)
    ref = PPO(env=LP.make_selected_env(), seed=3, agent_type="mlp", agent_config={"n_repeats": 4, "net_arch": []},
              run_name="ref", **kw)
    ref.train(total_timesteps=2048)
    mine = LP.LRPPO(env=LP.make_selected_env(), seed=3, agent_type="mlp", agent_config={"n_repeats": 4, "net_arch": []},
                    run_name="mine", eval_every=1000, eval_episodes=3, diag_every=500, **kw)
    mine.train(total_timesteps=2048)
    a = np.loadtxt("runs/ref.csv", delimiter=",", skiprows=1)
    b = np.loadtxt("runs/mine.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(a, b)                    # eval + diagnostics did not touch any RNG stream
    for p1, p2 in zip(ref.agent.parameters(), mine.agent.parameters()):
        assert torch.equal(p1, p2)
    assert len(mine.eval_rows) == 2 and len(mine.clip_log) == 8 * 4 * 4          # 8 iterations x 4 epochs x 4 minibatches


# ---------------------------------------------------------------------------------------------------- the collapse
@pytest.mark.parametrize("R", [4, 8, 32])
def test_collapsed_agent_step0_and_rng(R):
    torch.set_default_dtype(torch.float64)
    re_ = _ppo(f"ppo_reference_reuse{R}_clip", seed=5, f64=True)
    co = _ppo(f"ppo_reference_lr_x{R}_clip", seed=5, f64=True)
    # same initial network before the surgery means same RNG at the end of the construction
    assert LP.reuse_of(co.agent) == 1 and LP.reuse_of(re_.agent) == R
    x = torch.randn(64, 3, dtype=torch.float64)
    with torch.no_grad():
        d_a = (re_.agent.actor(re_.agent.input_reuse(x)) - co.agent.actor(x)).abs().max().item()
        d_c = (re_.agent.get_value(x) - co.agent.get_value(x)).abs().max().item()
    assert d_a < 1e-12 and d_c < 1e-12, (d_a, d_c)
    np.testing.assert_allclose(_effective(re_), _effective(co), atol=1e-12)
    a, b = re_, co
    assert a.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    assert b.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4 * R)
    assert b.optimizer.param_groups[1]["lr"] == pytest.approx(2.5e-4)
    assert len(b.optimizer.param_groups[0]["params"]) == 2        # actor[0].weight and critic[0].weight only
    assert all(p.dim() == 2 for p in b.optimizer.param_groups[0]["params"])
    assert b.optimizer.param_groups[0]["params"][0] is b.agent.actor[0].weight
    assert b.optimizer.param_groups[0]["params"][1] is b.agent.critic[0].weight
    assert len(b.optimizer.param_groups[1]["params"]) == 2        # the two biases; no other parameter exists


def test_collapse_does_not_consume_torch_rng():
    ppo = _ppo("ppo_reference_reuse8_clip", seed=2)
    state = torch.get_rng_state().clone()
    LP.collapse_agent_(ppo.agent)
    assert torch.equal(state, torch.get_rng_state())


def test_lr_annealing_scales_every_group():
    ppo = _ppo("ppo_reference_lr_x8_clip", seed=1, steps=512)          # 4 iterations of 128 steps: last one has frac 0.25
    g0, g1 = ppo.optimizer.param_groups
    assert g0["lr"] == pytest.approx(0.25 * 2.5e-4 * 8)
    assert g1["lr"] == pytest.approx(0.25 * 2.5e-4)


@pytest.mark.parametrize("arm,R", [("ppo_reference_lr_x8_clip", 8), ("ppo_reference_lr_x32_noclip", 32), ("ppo_reference_lr_x8_modified_clip", 8)])
def test_head_lr_is_R_times_the_others_at_every_update(arm, R):
    """Addendum 1: along the WHOLE trajectory (every update), lr(head group) == R * lr(other group), and both anneal."""
    seen = []

    def hook(p):
        gs = p.optimizer.param_groups
        seen.append((gs[0]["lr"], gs[1]["lr"]))

    ppo = _ppo(arm, seed=1, steps=0)
    ppo.iter_hook = hook
    ppo.train(total_timesteps=1024)                              # 8 iterations of 128 steps
    assert len(seen) == 8
    for k, (lh, lo) in enumerate(seen):
        frac = 1.0 - k / 8
        assert lo == pytest.approx(frac * 2.5e-4, rel=1e-12)
        assert lh == pytest.approx(R * lo, rel=1e-12)
    assert seen[-1][1] < seen[0][1] / 4                         # the base group really decays


def test_reuse_arms_anneal_their_single_group():
    seen = []
    ppo = _ppo("ppo_reference_reuse8_clip", seed=1, steps=0)
    ppo.iter_hook = lambda p: seen.append([g["lr"] for g in p.optimizer.param_groups])
    ppo.train(total_timesteps=512)
    assert all(len(s) == 1 for s in seen)
    assert [s[0] for s in seen] == pytest.approx([2.5e-4 * (1 - k / 4) for k in range(4)], rel=1e-12)


# --------------------------------------------------------------------------- equivalence of paths (P7, P8), float64
def _path(arm, steps, **kw):
    ppo = _ppo(arm, seed=4, steps=steps, f64=True, **kw)
    return ppo, _effective(ppo), np.asarray(ppo.clip_log)


@pytest.mark.parametrize("R", [8, 32])
def test_noclip_reuse_equals_lrx(R):
    re_, v_re, log_re = _path(f"ppo_reference_reuse{R}_noclip", 768)
    co, v_co, log_co = _path(f"ppo_reference_lr_x{R}_noclip", 768)
    assert np.abs(v_re - v_co).max() < 1e-9, np.abs(v_re - v_co).max()
    # the unmodified gradient norm of the collapsed net is smaller; for the reuse net it is the norm the rule would see
    assert log_re.shape == log_co.shape
    assert log_re[:, 0].mean() > log_co[:, 0].mean()


@pytest.mark.parametrize("R", [8, 32])
def test_modified_clip_reuse_equals_lrx_with_clipping_active(R):
    re_, v_re, log_re = _path(f"ppo_reference_reuse{R}_clip", 768)
    co, v_co, log_co = _path(f"ppo_reference_lr_x{R}_modified_clip", 768)
    assert log_re[:, 2].mean() > 0.2, "the test needs the clipping to fire often"
    assert np.abs(v_re - v_co).max() < 1e-9, np.abs(v_re - v_co).max()
    np.testing.assert_allclose(log_re[:, 0], log_co[:, 0], rtol=1e-8)        # same norm, update by update
    np.testing.assert_allclose(log_re[:, 1], log_co[:, 1], rtol=1e-8)


def test_usual_clip_rule_does_not_reproduce_reuse():
    re_, v_re, log_re = _path("ppo_reference_reuse32_clip", 768)
    co, v_co, log_co = _path("ppo_reference_lr_x32_clip", 768)
    assert np.abs(v_re - v_co).max() > 1e-5                    # without the modification the paths part


# ------------------------------------------------------------------------------------------ clipping diagnostic
def test_clip_diagnostic_matches_the_rule():
    ppo = _ppo("ppo_reference_reuse8_clip", seed=1, steps=512)
    log = np.asarray(ppo.clip_log)
    norms, factors, fired = log[:, 0], log[:, 1], log[:, 2].astype(bool)
    np.testing.assert_allclose(factors, np.minimum(1.0, 0.5 / (norms + 1e-6)))
    assert (fired == (norms > 0.5)).all()
    s = ppo.clip_summary()
    assert s["n_updates"] == len(log) and s["clip_fraction"] == pytest.approx(fired.mean())
    assert s["median_factor"] == pytest.approx(float(np.median(factors)))
    # no clipping: gradients untouched, but the would-be diagnostic is still recorded
    nc = _ppo("ppo_reference_reuse8_noclip", seed=1, steps=512)
    assert len(nc.clip_log) == len(log) and nc.clip_summary()["mode"] == "none"
    # the first update sees the same gradient with or without clipping
    assert nc.clip_log[0][0] == pytest.approx(log[0, 0], rel=1e-5)


def test_clipping_actually_applies_the_factor():
    ppo = _ppo("ppo_reference_reuse8_clip", seed=1, steps=0)
    p = ppo.agent.actor[0].weight
    p.grad = torch.full_like(p, 1.0)
    for q in ppo.agent.parameters():
        if q.grad is None:
            q.grad = torch.zeros_like(q)
    n0 = float(torch.sqrt(sum((q.grad ** 2).sum() for q in ppo.agent.parameters())))
    ppo._norm_and_clip()
    n1 = float(torch.sqrt(sum((q.grad ** 2).sum() for q in ppo.agent.parameters())))
    assert n0 > 0.5 and n1 == pytest.approx(0.5, rel=1e-4)
    assert ppo.clip_log[-1][0] == pytest.approx(n0, rel=1e-5) and ppo.clip_log[-1][2]


def test_modified_norm_weights_the_v_layers_by_R():
    ppo = _ppo("ppo_reference_lr_x8_modified_clip", seed=1, steps=0)
    for q in ppo.agent.parameters():
        q.grad = torch.zeros_like(q)
    ppo.agent.actor[0].weight.grad.fill_(0.1)
    ppo.agent.actor[0].bias.grad.fill_(0.1)
    v_sq = float((ppo.agent.actor[0].weight.grad ** 2).sum())
    b_sq = float((ppo.agent.actor[0].bias.grad ** 2).sum())
    ppo._norm_and_clip()
    assert ppo.clip_log[-1][0] == pytest.approx(np.sqrt(8 * v_sq + b_sq), rel=1e-6)
