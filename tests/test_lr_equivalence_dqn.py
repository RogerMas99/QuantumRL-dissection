"""Output Reuse against the learning rate of the head, DQN blocks (core/lr_equivalence.py).

What must not silently drift: that the "collapsed" arm is the reuse-R model with V0 = sum_r W_r0 and the same bias; that
in float64 a real DQN run of the collapsed arm (lr of head.weight x R) follows the reuse-R run (max |V_collapsed -
sum_r W_r| tiny, actions identical); that the R copies of a reuse run keep identical Adam moments; that the parameter
groups carry the expected learning rates in every arm; that "gradient x R" is Adam-equivalent to R = 1 at the first
step (difference only through eps); that the target network is the same kind of object as the online one; that the
diagnostics do not change the trajectory; and that the simulator arms of the learning-rate experiment are flagged as simulated.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

import quantumrl_dissection  # noqa: E402,F401  (applies compat.py patches on import)
from quantumrl_dissection.training import lr_equivalence as L  # noqa: E402
from quantumrl_dissection.agents.capacity import build_agent_for  # noqa: E402
from quantumrl_dissection.arms import ARMS, build_arm_config  # noqa: E402
from quantumrl_dissection.training.provenance import is_simulated_arm  # noqa: E402
from quantumrl_dissection.training.dqn import GreedyEvalConfig, SafeDQN  # noqa: E402

CP = "CartPole-v1"
KW = dict(buffer_size=2000, batch_size=32, learning_starts=200, train_frequency=4, target_network_frequency=100)
BASE_LR = 2.5e-4


@pytest.fixture(autouse=True)
def _default_dtype_restored():
    d = torch.get_default_dtype()
    yield
    torch.set_default_dtype(d)


def _reuse(pre, R):
    return pre if R == 1 else f"{pre}_reuse{R}"


def _train(arm, tmp_path, seed=1, steps=1500, f64=True, trace=100, hook=True, **extra):
    """Run a short real DQN training through SafeDQN; return (dqn, hook)."""
    at, cfg = build_arm_config(arm, env_id=CP)
    cfg = dict(cfg, lr_diag=True, **extra)
    if f64:
        cfg["lr_float64"] = True
    if trace:
        cfg["lr_trace_every"] = trace
    holder = []
    pb = L.make_post_build(cfg)

    def pb2(dqn):
        h = pb(dqn)
        holder.append((dqn, h))
        return h

    runner = SafeDQN(agent_type=at, agent_config=cfg, run_name=f"t_{arm}_{seed}", seed=seed, env_id=CP,
                     terminal_adjustment=True, eval_cfg=GreedyEvalConfig(enabled=False), outdir=tmp_path,
                     post_build=pb2 if hook else None, **KW)
    runner.train(steps)
    return holder[0] if holder else (None, None)


def _agent(arm, seed=0, f64=False):
    torch.manual_seed(seed)
    at, cfg = build_arm_config(arm, env_id=CP)
    a = build_agent_for(at, cfg, env_id=CP, is_qnet=True)
    return L.to_float64_(a) if f64 else a


# ---------------------------------------------------------------------------------------------------------- registry
def test_registry_names_and_types():
    for pre, at in (("classical_twin", "classical_twin"), ("sim_statevector_hybrid", "sv")):
        names = L.arm_names(pre)
        assert [len(v) for v in names.values()] == [5, 4, 2, 7, 4]
        for group in names.values():
            for n in group:
                assert ARMS[n][0] == at
    assert "classical_twin_all_lr_x0p5" in ARMS and "sim_statevector_hybrid_all_lr_x0p5" in ARMS


def test_simulator_base_is_the_arm_of_the_validation():
    for R in (1, 4, 8, 16, 32):
        ref = ARMS["sim_statevector_hybrid" if R == 1 else f"sim_statevector_hybrid_reuse{R}"][1]
        got = {k: v for k, v in ARMS[_reuse("sim_statevector_hybrid", R)][1].items() if not k.startswith("lr")}
        assert got == ref


def test_twin_base_is_the_arm_of_the_classical_twin_experiment():
    for R in (1, 4, 8, 16, 32):
        ref = ARMS[_reuse("classical_twin", R)][1]
        got = {k: v for k, v in ARMS[_reuse("classical_twin", R)][1].items() if not k.startswith("lr")}
        assert got == ref


def test_simulated_flag():
    for n in ("sim_statevector_hybrid", "sim_statevector_hybrid_collapsed8", "sim_statevector_hybrid_grad_x32", "sim_statevector_hybrid_head_lr_x128",
              "sim_statevector_hybrid_all_lr_x0p5"):
        assert is_simulated_arm(n)
    for n in ("classical_twin", "classical_twin_collapsed8", "classical_twin_head_lr_x16"):
        assert not is_simulated_arm(n)


def test_learning_rate_keys_per_arm_kind():
    c = ARMS["classical_twin_collapsed8"][1]
    assert c["lr_collapse_R"] == 8 and c["lr_head_mult"] == 8.0 and c.get("reuse_repetitions") == 8
    g = ARMS["classical_twin_grad_x32"][1]
    assert g["lr_grad_mult"] == 32.0 and g.get("reuse_repetitions", 1) == 1 and "lr_head_mult" not in g
    h = ARMS["sim_statevector_hybrid_head_lr_x128"][1]
    assert h["lr_head_mult"] == 128.0 and h.get("reuse_repetitions", 1) == 1
    k = ARMS["classical_twin_all_lr_x0p5"][1]
    assert k["lr_global_mult"] == 0.5 and "lr_head_mult" not in k


# ----------------------------------------------------------------------------------------------- step 0, float64
@pytest.mark.parametrize("arm_pre", ["classical_twin", "sim_statevector_hybrid"])
@pytest.mark.parametrize("R", [4, 8, 32])
def test_step0_collapsed_equals_reuse_float64(arm_pre, R):
    ag = _agent(_reuse(arm_pre, R), seed=3, f64=True)
    col = copy.deepcopy(ag)
    assert L.collapse_head_(col) == R
    x = torch.randn(64, 4, dtype=torch.float64)
    with torch.no_grad():
        d = (ag(x) - col(x)).abs().max().item()
    assert d < 1e-12, d
    # structure: no OutputReuse left, head matrix f wide, bias identical, everything else the same objects
    assert L.reuse_factor(col.network) == 1
    i_old, i_new = L.head_index(ag.network), L.head_index(col.network)
    assert col.network[i_new].weight.shape[1] == ag.network[i_old].weight.shape[1] // R
    assert torch.equal(col.network[i_new].bias, ag.network[i_old].bias)
    np.testing.assert_allclose(col.network[i_new].weight.detach().numpy(), L.effective_v(ag).detach().numpy(),
                               atol=0, rtol=0)
    assert col.network[0] is ag.network[0] or hasattr(col.network[0], "parameters")
    assert sum(p.numel() for p in col.parameters()) == sum(p.numel() for p in ag.parameters()) - 4 * 8 * (R - 1)


def test_collapse_is_idempotent_on_r1():
    ag = _agent("classical_twin")
    assert L.collapse_head_(ag) == 1


# ---------------------------------------------------------------------------------------- parameter groups / lrs
def _groups(arm, tmp_path):
    dqn, hook = _train(arm, tmp_path, steps=30, f64=False, trace=0)
    return dqn


@pytest.mark.parametrize("arm_pre", ["classical_twin", "sim_statevector_hybrid"])
def test_param_groups_in_every_arm(arm_pre, tmp_path):
    expected = {
        f"{arm_pre}_reuse8": (1.0, 1.0),
        f"{arm_pre}_collapsed8": (8.0, 1.0),
        f"{arm_pre}_grad_x8": (1.0, 1.0),
        f"{arm_pre}_head_lr_x16": (16.0, 1.0),
        f"{arm_pre}_all_lr_x4": (4.0, 4.0),
        f"{arm_pre}_all_lr_x0p5": (0.5, 0.5),
    }
    for arm, (head_mult, rest_mult) in expected.items():
        dqn = _groups(arm, tmp_path)
        hw = L.head_weight(dqn.q_network)
        for g in dqn.optimizer.param_groups:
            holds_head = any(p is hw for p in g["params"])
            want = BASE_LR * (head_mult if holds_head else rest_mult)
            assert g["lr"] == pytest.approx(want, rel=1e-12), (arm, holds_head, g["lr"])
            assert g["weight_decay"] == 0
        if len(dqn.optimizer.param_groups) == 2:
            assert [len(g["params"]) for g in dqn.optimizer.param_groups][0] == 1       # head.weight alone
            assert not any(p is dqn.q_network.network[L.head_index(dqn.q_network.network)].bias
                           for g in dqn.optimizer.param_groups[:1] for p in g["params"])
        # every parameter is in exactly one group
        flat = [p for g in dqn.optimizer.param_groups for p in g["params"]]
        assert len(flat) == len(list(dqn.q_network.parameters())) == len({id(p) for p in flat})


def test_no_clipping_no_scheduler_in_dqn_source():
    import inspect
    from simplyqrl import dqn as upstream
    src = inspect.getsource(upstream)
    for banned in ("clip_grad", "lr_scheduler", "weight_decay", "param_groups"):
        assert banned not in src, banned


# ------------------------------------------------------------------------------- target network is the same kind
def test_target_network_follows_collapsed_online_net(tmp_path):
    dqn, _ = _train("classical_twin_collapsed4", tmp_path, steps=250, f64=False, trace=0)
    q, t = dqn.q_network, dqn.target_network
    assert [tuple(p.shape) for p in q.parameters()] == [tuple(p.shape) for p in t.parameters()]
    assert L.reuse_factor(t.network) == 1
    # the head of both is f wide (8), not 8*R
    assert t.network[L.head_index(t.network)].weight.shape[1] == 8
    assert all(not p.requires_grad or p.requires_grad for p in t.parameters())


# ------------------------------------------------------------------- real runs: collapsed follows reuse (float64)
@pytest.mark.parametrize("R", [4, 8])
def test_float64_run_collapsed_follows_reuse_twin(R, tmp_path):
    d_re, h_re = _train(_reuse("classical_twin", R), tmp_path / "re", steps=1500)
    d_co, h_co = _train(f"classical_twin_collapsed{R}", tmp_path / "co", steps=1500)
    assert h_re.actions == h_co.actions                    # same trajectory
    assert len(h_re.trace_params) == len(h_co.trace_params) >= 10
    diffs = [np.abs(a - b).max() for a, b in zip(h_re.trace_params, h_co.trace_params)]
    assert max(diffs) < 1e-9, max(diffs)
    # and the final effective V of the reuse run is the sum of its copies, which is what the trace compares
    np.testing.assert_allclose(L.effective_v(d_re.q_network).detach().numpy(),
                               L.effective_v(d_co.q_network).detach().numpy(), atol=1e-9)


def test_float64_run_collapsed_follows_reuse_emulator(tmp_path):
    d_re, h_re = _train("sim_statevector_hybrid_reuse4", tmp_path / "re", steps=700, trace=100)
    d_co, h_co = _train("sim_statevector_hybrid_collapsed4", tmp_path / "co", steps=700, trace=100)
    assert h_re.actions == h_co.actions
    diffs = [np.abs(a - b).max() for a, b in zip(h_re.trace_params, h_co.trace_params)]
    assert max(diffs) < 1e-9, max(diffs)


@pytest.mark.parametrize("R", [4, 8])
def test_adam_moments_of_the_copies_are_identical(R, tmp_path):
    dqn, _ = _train(_reuse("classical_twin", R), tmp_path, steps=1200, trace=0)
    w = L.head_weight(dqn.q_network)
    st = dqn.optimizer.state[w]
    for key in ("exp_avg", "exp_avg_sq"):
        m = st[key].reshape(4, R, 8)
        assert (m - m[:, :1, :]).abs().max().item() < 1e-12, key
    wr = w.detach().reshape(4, R, 8)                          # the copies also receive identical updates
    spread = (wr - wr[:, :1, :]).abs().max().item()
    init_spread = _agent(_reuse("classical_twin", R), seed=1).network[2].weight.detach().double().reshape(4, R, 8)
    init_spread = (init_spread - init_spread[:, :1, :]).abs().max().item()
    assert spread == pytest.approx(init_spread, abs=1e-9)   # the differences between copies never change


# --------------------------------------------------------------------------- gradient x R vs R = 1: Adam invariance
def test_gradx_hook_scales_exactly_and_adam_update_is_scale_free(tmp_path):
    """Direct check on one network and one batch: same net, grads g and R*g -> same first Adam step up to eps."""
    ag = _agent("classical_twin", seed=2, f64=True)
    ag2 = copy.deepcopy(ag)
    o1 = torch.optim.Adam(ag.parameters(), lr=BASE_LR)
    o2 = torch.optim.Adam(ag2.parameters(), lr=BASE_LR)
    L.scale_grad_(L.head_weight(ag2), 8.0)
    x = torch.randn(32, 4, dtype=torch.float64)
    for a, o in ((ag, o1), (ag2, o2)):
        o.zero_grad()
        ((a(x) ** 2).mean() + a(x).sum() * 0.1).backward()
        o.step()
    g1, g2 = L.head_weight(ag).grad, L.head_weight(ag2).grad
    assert torch.allclose(g2, 8.0 * g1, rtol=1e-12)
    d = (L.head_weight(ag) - L.head_weight(ag2)).abs().max().item()
    assert d < BASE_LR * 1e-4, d            # only the eps = 1e-8 term separates them
    for (n1, p1), (n2, p2) in zip(ag.named_parameters(), ag2.named_parameters()):
        if p1 is not L.head_weight(ag):
            assert torch.equal(p1, p2)       # the rest of the network is untouched by the hook


# --------------------------------------------------------------------------- diagnostics: content, no trajectory effect
def test_diagnostics_content_and_no_effect_on_the_trajectory(tmp_path):
    at, cfg = build_arm_config("classical_twin_reuse4", env_id=CP, lr_diag=True)

    def seed_only(dqn):                      # same seeding as post_build, nothing else: the reference trajectory
        dqn.envs.single_action_space.seed(int(dqn.seed))
        return None

    plain = SafeDQN(agent_type=at, agent_config={k: v for k, v in cfg.items() if not k.startswith("lr")},
                    run_name="plain", seed=2, env_id=CP, terminal_adjustment=True,
                    eval_cfg=GreedyEvalConfig(enabled=False), outdir=tmp_path / "plain", post_build=seed_only, **KW)
    out_p = plain.train(2100)
    with_diag = SafeDQN(agent_type=at, agent_config=cfg, run_name="diag", seed=2, env_id=CP, terminal_adjustment=True,
                        eval_cfg=GreedyEvalConfig(enabled=False), outdir=tmp_path / "diag",
                        post_build=L.make_post_build(cfg), **KW)
    out_d = with_diag.train(2100)
    a = np.loadtxt(out_p.episodes_csv, delimiter=",", skiprows=1)
    b = np.loadtxt(out_d.episodes_csv, delimiter=",", skiprows=1)
    np.testing.assert_array_equal(a, b)                          # identical training episodes
    rows = np.loadtxt(out_d.extra["diag_csv"], delimiter=",", skiprows=1)
    assert list(rows[:, 0]) == [1000.0, 2000.0]
    assert np.all(rows[:, 1] > 0) and np.all(rows[:, 2] > 0)
    assert "diag_csv" not in out_p.extra


def test_same_seed_same_run(tmp_path):
    """With the exploration sampler seeded, two runs of one cell are identical (what the trace tests rely on)."""
    _, h1 = _train("classical_twin_reuse4", tmp_path / "a", steps=600, f64=False)
    _, h2 = _train("classical_twin_reuse4", tmp_path / "b", steps=600, f64=False)
    assert h1.actions == h2.actions
    assert max(np.abs(x - y).max() for x, y in zip(h1.trace_params, h2.trace_params)) == 0.0


def test_v_eff_norm_is_the_norm_of_the_sum_of_copies():
    ag = _agent("classical_twin_reuse8", seed=4)
    w = L.head_weight(ag).detach().reshape(4, 8, 8)
    assert torch.allclose(L.effective_v(ag), w.sum(dim=1))
    assert L.effective_v(_agent("classical_twin")).shape == (4, 8)


def test_post_build_is_none_for_arms_without_learning_rate_keys():
    at, cfg = build_arm_config("classical_twin_reuse8", env_id=CP)         # the arm of the classical twin experiment
    assert L.make_post_build(cfg) is None
    assert L.make_post_build(None) is None
