"""
Capacity-matched classical control arm.

The problem
-----------
The paper's classical control (section 3.4A) is the hybrid arm MINUS the PQC:
same input dimensionality, same linear readout, comparable parameter count. The
point is that the only difference between arms is the circuit, so any gap is
attributable to it.

That logic silently assumes both arms are ALIVE. Under PPO they are. Under DQN
the classical arm dies at ~9.7 (below a random policy) while the hybrid arm
still learns, and a comparison between a live agent and a dead one says nothing
about the PQC: the death may come from DQN interacting with a linear Q-function
(deadly triad, bootstrapping to ~1/(1-gamma) from a couple dozen parameters)
rather than from the absence of a circuit.

The remedy is NOT to give the classical arm a large MLP - that destroys the
attribution the whole design exists to protect. It is to replace the PQC with a
classical block of THE SAME PARAMETER COUNT, so the comparison stays honest and
the arm has a chance to learn.

This module measures the PQC budget empirically (never analytically - the
formula depends on circuit template details) and solves for the hidden width
that spends the same budget.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# torch is imported lazily: match_hidden_width is pure arithmetic and must be
# testable without a quantum stack installed. See tests/conftest.py.


def count_trainable(module: "nn.Module") -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_agent_for(
    agent_type: str,
    agent_config: Dict[str, Any],
    env_id: str = "CartPole-v1",
    is_qnet: bool = True,
) -> "nn.Module":
    """Construct an agent without training it. Cheap; used for accounting.

    An arm that selects components of the observation in the environment (``_env_select``) is built against that environment.
    """
    import gymnasium as gym

    from simplyqrl.agents import build_agent  # patched by compat.py

    select = (agent_config or {}).get("_env_select")
    if select is not None:
        from quantumrl_dissection.environments import make_selected_env
        env = make_selected_env(env_id, select)()
    else:
        env = gym.make(env_id)
    try:
        obs_shape = env.observation_space.shape
        n_actions = int(env.action_space.n)
        return build_agent(agent_type, obs_shape, n_actions, agent_config,
                           is_qnet=is_qnet)
    finally:
        env.close()


def pqc_parameter_budget(hybrid_config: Dict[str, Any],
                         env_id: str = "CartPole-v1") -> Dict[str, int]:
    """Split a hybrid agent's parameters into quantum and classical head.

    The quantum budget is what the capacity-matched classical arm must spend.
    """
    agent = build_agent_for("hybrid", hybrid_config, env_id=env_id, is_qnet=True)
    total = count_trainable(agent)

    quantum = 0
    for name, param in agent.named_parameters():
        if not param.requires_grad:
            continue
        # PennyLane TorchLayer registers its weights under the qlayer module;
        # everything else is the classical head.
        if "q_layer" in name or "qlayer" in name or "weights" in name.split(".")[-1]:
            quantum += param.numel()

    return {"total": total, "quantum": quantum, "classical_head": total - quantum}


def match_hidden_width(
    target_params: int,
    in_dim: int,
    out_dim: int,
    max_width: int = 4096,
) -> Tuple[int, int]:
    """Smallest hidden width whose block spends closest to `target_params`.

    The block is Linear(in_dim, h) + activation + Linear(h, out_dim), i.e. the
    classical stand-in for "PQC then linear readout".

    Returns (width, actual_param_count).
    """
    if target_params <= 0:
        raise ValueError("target_params must be positive")

    def cost(h: int) -> int:
        return in_dim * h + h + h * out_dim + out_dim

    best_h, best_err = 1, abs(cost(1) - target_params)
    for h in range(1, max_width + 1):
        err = abs(cost(h) - target_params)
        if err < best_err:
            best_h, best_err = h, err
        if cost(h) > target_params * 3:
            break
    return best_h, cost(best_h)


def capacity_ladder(
    arms: Dict[str, Tuple[str, Dict[str, Any]]],
    env_id: str = "CartPole-v1",
) -> List[Dict[str, Any]]:
    """Parameter accounting for a set of arms. Seconds; trains nothing.

    Run this before committing compute. If an arm does not build what you think
    it builds, you want to know now and not after twelve runs.
    """
    rows = []
    for label, (agent_type, cfg) in arms.items():
        agent = build_agent_for(agent_type, cfg, env_id=env_id, is_qnet=True)
        n = count_trainable(agent)
        row = {"arm": label, "agent_type": agent_type, "trainable_params": n,
               "structure": str(agent)}
        if agent_type == "hybrid":
            row.update({"pqc_" + k: v for k, v in
                        pqc_parameter_budget(cfg, env_id=env_id).items()})
        rows.append(row)
    return rows
