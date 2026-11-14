"""Output reuse followed by a linear layer is the same as multiplying the learning rate of the weights of that layer by R.

With R copies of the features, the output depends on the weights through V = sum_r W_r. The model with the head collapsed to V and
the learning rate of V times R follows the same path in float64, under DQN and under PPO without clipping or with the modified rule.
"""
import copy

import numpy as np
import pytest
import torch
from simplyqrl.ppo import PPO

from quantumrl_dissection import learning_rate as L
from quantumrl_dissection.arms import build_arm_config
from quantumrl_dissection.models import build_agent_for, make_selected_env
from quantumrl_dissection.training import GreedyEvalConfig, SafeDQN

CP = "CartPole-v1"
KW = dict(buffer_size=2000, batch_size=32, learning_starts=200, train_frequency=4, target_network_frequency=100)
BASE_LR = 2.5e-4


@pytest.fixture(autouse=True)
def _default_dtype_restored(monkeypatch, tmp_path):
    dtype = torch.get_default_dtype()
    monkeypatch.chdir(tmp_path)
    yield
    torch.set_default_dtype(dtype)


def _reuse(prefix, R):
    return prefix if R == 1 else f"{prefix}_reuse{R}"


def _agent(arm, seed=0, float64=False):
    torch.manual_seed(seed)
    agent_type, cfg = build_arm_config(arm, env_id=CP)
    agent = build_agent_for(agent_type, cfg, env_id=CP, is_qnet=True)
    return L.to_float64_(agent) if float64 else agent


def _train(arm, tmp_path, steps=1500, float64=True, trace=100, seed=1):
    """Train through SafeDQN; returns (dqn, hook)."""
    agent_type, cfg = build_arm_config(arm, env_id=CP)
    cfg = dict(cfg, lr_seed_sampler=True, lr_float64=float64, lr_trace_every=trace)
    runner = SafeDQN(agent_type=agent_type, agent_config=cfg, run_name=f"t_{arm}", seed=seed, env_id=CP, terminal_adjustment=True,
                     eval_cfg=GreedyEvalConfig(enabled=False), outdir=tmp_path, post_build=L.make_post_build(cfg), **KW)
    runner.train(steps)
    return runner.dqn, runner.hook


# --- DQN ----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("prefix", ["classical_twin", "sim_statevector_hybrid"])
@pytest.mark.parametrize("R", [4, 8])
def test_collapsed_model_computes_the_same_function(prefix, R):
    agent = _agent(_reuse(prefix, R), seed=3, float64=True)
    collapsed = copy.deepcopy(agent)
    assert L.collapse_head_(collapsed) == R
    x = torch.randn(64, 4, dtype=torch.float64)
    with torch.no_grad():
        assert (agent(x) - collapsed(x)).abs().max().item() < 1e-12
    assert L.reuse_factor(collapsed.network) == 1
    np.testing.assert_array_equal(L.head_weight(collapsed).detach().numpy(), L.effective_v(agent).detach().numpy())


@pytest.mark.parametrize("prefix,R,steps", [("classical_twin", 4, 1500), ("classical_twin", 8, 1500), ("sim_statevector_hybrid", 4, 700)])
def test_collapsed_run_follows_the_reuse_run(prefix, R, steps, tmp_path):
    reuse_dqn, reuse = _train(_reuse(prefix, R), tmp_path / "reuse", steps)
    collapsed_dqn, collapsed = _train(f"{prefix}_collapsed{R}", tmp_path / "collapsed", steps)
    assert reuse.actions == collapsed.actions
    assert len(reuse.trace_params) == len(collapsed.trace_params) >= 6
    assert max(np.abs(a - b).max() for a, b in zip(reuse.trace_params, collapsed.trace_params)) < 1e-9
    np.testing.assert_allclose(L.effective_v(reuse_dqn.q_network).detach().numpy(),
                               L.effective_v(collapsed_dqn.q_network).detach().numpy(), atol=1e-9)


def test_the_copies_keep_identical_adam_moments(tmp_path):
    dqn, _ = _train("classical_twin_reuse4", tmp_path, steps=1200, trace=0)
    state = dqn.optimizer.state[L.head_weight(dqn.q_network)]
    for key in ("exp_avg", "exp_avg_sq"):
        m = state[key].reshape(4, 4, 8)
        assert (m - m[:, :1, :]).abs().max().item() < 1e-12, key


def test_scaling_the_gradient_does_not_change_the_adam_update():
    agent = _agent("classical_twin", seed=2, float64=True)
    scaled = copy.deepcopy(agent)
    L.scale_grad_(L.head_weight(scaled), 8.0)
    x = torch.randn(32, 4, dtype=torch.float64)
    for model in (agent, scaled):
        optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
        optimizer.zero_grad()
        ((model(x) ** 2).mean() + model(x).sum() * 0.1).backward()
        optimizer.step()
    assert (L.head_weight(agent) - L.head_weight(scaled)).abs().max().item() < BASE_LR * 1e-4


@pytest.mark.parametrize("arm,head,rest", [("classical_twin_reuse8", 1, 1), ("classical_twin_collapsed8", 8, 1),
                                           ("classical_twin_head_lr_x16", 16, 1), ("classical_twin_all_lr_x4", 4, 4)])
def test_parameter_groups_carry_the_learning_rates(arm, head, rest, tmp_path):
    dqn, _ = _train(arm, tmp_path, steps=30, float64=False, trace=0)
    head_weight = L.head_weight(dqn.q_network)
    for group in dqn.optimizer.param_groups:
        holds_head = any(p is head_weight for p in group["params"])
        assert group["lr"] == pytest.approx(BASE_LR * (head if holds_head else rest), rel=1e-12)
        assert group["weight_decay"] == 0


def test_same_seed_same_run(tmp_path):
    _, a = _train("classical_twin_reuse4", tmp_path / "a", steps=600, float64=False)
    _, b = _train("classical_twin_reuse4", tmp_path / "b", steps=600, float64=False)
    assert a.actions == b.actions
    assert max(np.abs(x - y).max() for x, y in zip(a.trace_params, b.trace_params)) == 0.0


# --- PPO ----------------------------------------------------------------------------------------------------------------------------
SMALL = dict(L.REFERENCE_PPO_CONFIG, num_steps=128, num_minibatches=4)


def _ppo(arm, seed=4, steps=0, float64=False):
    if float64:
        torch.set_default_dtype(torch.float64)
    a = L.PPO_ARMS[arm]
    ppo = L.LRPPO(env=make_selected_env(), seed=seed, agent_type="mlp", agent_config={"n_repeats": a.R, "net_arch": []},
                  run_name=f"{arm}_{seed}", head_lr_mult=float(a.R) if a.collapse else 1.0, collapse=a.collapse,
                  clip_mode=a.clip_mode, clip_R=a.R, float64=float64, **SMALL)
    if steps:
        ppo.train(total_timesteps=steps)
    return ppo


def _effective(ppo):
    """V and the bias of the actor and of the critic, flattened."""
    out = []
    for seq in (ppo.agent.actor, ppo.agent.critic):
        w, b = seq[0].weight.detach().double(), seq[0].bias.detach().double()
        R = L.reuse_of(ppo.agent)
        o, i = w.shape
        out += [(w.reshape(o, R, i // R).sum(dim=1) if R > 1 else w).reshape(-1), b.reshape(-1)]
    return torch.cat(out).numpy()


def test_ppo_with_the_defaults_is_the_library_ppo():
    kw = dict(SMALL, num_steps=256)
    config = {"n_repeats": 4, "net_arch": []}
    ref = PPO(env=make_selected_env(), seed=3, agent_type="mlp", agent_config=config, run_name="ref", **kw)
    ref.train(total_timesteps=2048)
    mine = L.LRPPO(env=make_selected_env(), seed=3, agent_type="mlp", agent_config=config, run_name="mine", eval_every=1000,
                   eval_episodes=3, **kw)
    mine.train(total_timesteps=2048)
    np.testing.assert_array_equal(np.loadtxt("runs/ref.csv", delimiter=",", skiprows=1), np.loadtxt("runs/mine.csv", delimiter=",", skiprows=1))
    assert all(torch.equal(p, q) for p, q in zip(ref.agent.parameters(), mine.agent.parameters()))


@pytest.mark.parametrize("R", [8, 32])
def test_without_clipping_the_reuse_run_equals_the_scaled_learning_rate(R):
    reuse = _effective(_ppo(f"ppo_reference_reuse{R}_noclip", steps=768, float64=True))
    collapsed = _effective(_ppo(f"ppo_reference_lr_x{R}_noclip", steps=768, float64=True))
    assert np.abs(reuse - collapsed).max() < 1e-9


@pytest.mark.parametrize("R", [8, 32])
def test_with_the_modified_clipping_rule_the_reuse_run_equals_the_scaled_learning_rate(R):
    reuse = _ppo(f"ppo_reference_reuse{R}_clip", steps=768, float64=True)
    collapsed = _ppo(f"ppo_reference_lr_x{R}_modified_clip", steps=768, float64=True)
    log_reuse, log_collapsed = np.asarray(reuse.clip_log), np.asarray(collapsed.clip_log)
    assert log_reuse[:, 2].mean() > 0.2, "the clipping must fire often for the test to mean anything"
    assert np.abs(_effective(reuse) - _effective(collapsed)).max() < 1e-9
    np.testing.assert_allclose(log_reuse[:, :2], log_collapsed[:, :2], rtol=1e-8)


def test_the_usual_clipping_rule_does_not_reproduce_reuse():
    reuse = _effective(_ppo("ppo_reference_reuse32_clip", steps=768, float64=True))
    collapsed = _effective(_ppo("ppo_reference_lr_x32_clip", steps=768, float64=True))
    assert np.abs(reuse - collapsed).max() > 1e-5
