"""Output reuse against the learning rate: the variants of DQN (collapsed head, scaled head learning rate, scaled gradient)
and PPO with a per-group learning rate and modified gradient clipping.

With R copies of the features z reaching a linear head, q = [W_1 | ... | W_R][z; ...; z] + b depends on the copies only through
V = sum_r W_r. With SGD, momentum or Adam, no global gradient clipping and no weight decay, training is the same as without reuse
(head V) if the learning rate of V is multiplied by R and V^0 = sum_r W_r^0. The functions here act on a constructed ``DQN``
between ``__init__`` and ``train`` (``post_build``); the library is not modified.

Keys of the configuration of an arm (the agent classes ignore keys they do not know):

    lr_seed_sampler          True  -> seed the exploration sampler with the seed of the run
    lr_collapse_R    int   -> replace the head of the reuse-R model by one with V^0 = sum_r W_r^0
    lr_head_mult     float -> learning rate of the head weight times this factor
    lr_grad_mult     float -> gradient of the head weight times this factor
    lr_global_mult   float -> learning rate of all the parameters times this factor
    lr_float64       True  -> network arithmetic in float64
    lr_trace_every   int   -> keep the effective parameters and the actions every that many steps
    lr_classical     True  -> classical models with reuse built in (``lr_classical_R`` copies)
"""
from __future__ import annotations

import copy
import csv
import math
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from simplyqrl.ppo import PPO
from tqdm import tqdm

from quantumrl_dissection.models import N_UNIQUE_WIRES, REFERENCE_COMPONENTS, make_selected_env

KEYS = ("lr_seed_sampler", "lr_classical", "lr_collapse_R", "lr_head_mult", "lr_grad_mult", "lr_global_mult", "lr_float64")


def has_lr(cfg: Optional[Dict[str, Any]]) -> bool:
    return bool(cfg) and any(k in cfg for k in KEYS)


def head_index(net: nn.Sequential) -> int:
    """Index of the first top-level ``nn.Linear`` of the network: the layer that reads the replicated features."""
    for i, m in enumerate(net):
        if isinstance(m, nn.Linear):
            return i
    raise RuntimeError("no top-level nn.Linear in agent.network")


def reuse_factor(net: nn.Sequential) -> int:
    for m in net:
        if type(m).__name__ == "OutputReuse":
            return int(m.n_repeats)
    return 1


def head_weight(agent: nn.Module) -> nn.Parameter:
    return agent.network[head_index(agent.network)].weight


def effective_v(agent: nn.Module) -> torch.Tensor:
    """V = sum_r W_r of the head (the weight itself without reuse)."""
    net = agent.network
    w = net[head_index(net)].weight
    R = reuse_factor(net)
    out, inn = w.shape
    return w.reshape(out, R, inn // R).sum(dim=1) if R > 1 else w


def _sum_blocks(old: nn.Linear, R: int) -> nn.Linear:
    """Linear layer with V^0 = sum of the R column blocks of ``old`` and its bias; leaves the RNG stream where it was."""
    out, inn = old.weight.shape
    f = inn // R
    with torch.random.fork_rng(devices=[]):
        new = nn.Linear(f, out, bias=old.bias is not None).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight.reshape(out, R, f).sum(dim=1))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


def collapse_head_(agent: nn.Module) -> int:
    """Replace OutputReuse(R) + Linear(f*R -> out) by Linear(f -> out) with V^0 = sum_r W_r^0. Returns R."""
    net = agent.network
    R = reuse_factor(net)
    if R == 1:
        return 1
    i = head_index(net)
    new = _sum_blocks(net[i], R)
    agent.network = nn.Sequential(*[new if j == i else m for j, m in enumerate(net) if type(m).__name__ != "OutputReuse"])
    if hasattr(agent, "reuse_repetitions"):
        agent.reuse_repetitions = 1
    return R


def _split_adam(agent: nn.Module, head: nn.Parameter, base_lr: float, head_lr: float, rest_lr: float) -> torch.optim.Adam:
    rest = [p for p in agent.parameters() if p is not head]
    return torch.optim.Adam([{"params": [head], "lr": head_lr}, {"params": rest, "lr": rest_lr}], lr=base_lr)


def build_adam(agent: nn.Module, base_lr: float, head_mult: float = 1.0, global_mult: float = 1.0) -> torch.optim.Adam:
    """Adam as in the library, with two groups: the head weight (lr x global x head) and the rest (lr x global)."""
    return _split_adam(agent, head_weight(agent), base_lr, base_lr * global_mult * head_mult, base_lr * global_mult)


def scale_grad_(param: nn.Parameter, factor: float):
    return param.register_hook(lambda g: g * factor)


def _cast_inputs_float64(module, args):
    return tuple(a.to(torch.float64) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)


def to_float64_(agent: nn.Module) -> nn.Module:
    agent.double()
    agent.register_forward_pre_hook(_cast_inputs_float64)
    return agent


class TraceHook:
    """Records the actions and the effective parameters every ``trace_every`` steps; nothing here draws from a random stream."""

    def __init__(self, dqn, trace_every: int = 0):
        self.dqn, self.trace_every = dqn, trace_every
        self.actions: List[int] = []
        self.trace_steps: List[int] = []
        self.trace_params: List[np.ndarray] = []
        if trace_every:
            orig = dqn.envs.step

            def _step(actions, _orig=orig):
                self.actions.append(int(np.asarray(actions).reshape(-1)[0]))
                return _orig(actions)

            dqn.envs.step = _step

    def _flat_effective(self) -> np.ndarray:
        q = self.dqn.q_network
        net = q.network
        hi = head_index(net)
        parts = []
        for j, m in enumerate(net):
            if j == hi:
                parts.append(effective_v(q).detach().reshape(-1))
                if m.bias is not None:
                    parts.append(m.bias.detach().reshape(-1))
            else:
                parts += [p.detach().reshape(-1) for p in m.parameters()]
        return torch.cat(parts).double().cpu().numpy()

    def on_step(self, n: int) -> None:
        if self.trace_every and n % self.trace_every == 0:
            self.trace_steps.append(n)
            self.trace_params.append(self._flat_effective())


def head_linear(agent: nn.Module) -> nn.Linear:
    """The head of a classical model with reuse built in: ``agent.head`` (ceiling) or the first top-level Linear."""
    if isinstance(getattr(agent, "head", None), nn.Linear):
        return agent.head
    return agent.network[head_index(agent.network)]


def effective_v_blocks(agent: nn.Module, R: int) -> torch.Tensor:
    w = head_linear(agent).weight
    out, inn = w.shape
    return w.reshape(out, R, inn // R).sum(dim=1) if R > 1 else w


def collapse_blocks_(agent: nn.Module, R: int) -> None:
    """Make the model that replicates its features R times into the one without replication, with V^0 = sum of the blocks."""
    new = _sum_blocks(head_linear(agent), R)
    if isinstance(getattr(agent, "head", None), nn.Linear):
        if agent.n_qubits != R * N_UNIQUE_WIRES:
            raise RuntimeError("collapse expects n_qubits == R * 4 for the Fourier ceiling")
        agent.n_qubits = N_UNIQUE_WIRES
        agent.head = new
        return
    net = agent.network
    i = head_index(net)
    agent.network = nn.Sequential(*[new if j == i else m for j, m in enumerate(net)])
    ir = getattr(agent, "input_reuse", None)
    if ir is not None:
        if getattr(ir, "n_repeats", None) != R:
            raise RuntimeError("input_reuse.n_repeats does not match lr_classical_R")
        ir.n_repeats = 1


class BlocksTraceHook(TraceHook):
    """``TraceHook`` for models whose head is not the first Linear of ``network`` or whose features are replicated outside it."""

    def __init__(self, dqn, R: int, trace_every: int = 0):
        self.R = R
        super().__init__(dqn, trace_every)

    def _flat_effective(self):
        q = self.dqn.q_network
        hw, parts = head_linear(q), []
        for p in q.parameters():
            parts.append((effective_v_blocks(q, self.R) if p is hw.weight else p).detach().reshape(-1))
        return torch.cat(parts).double().cpu().numpy()


def make_post_build(cfg: Optional[Dict[str, Any]]):
    """``post_build(dqn) -> hook`` for an arm with an lr_* key, or None (then the pipeline is the unchanged one)."""
    if not has_lr(cfg):
        return None
    collapse = int(cfg.get("lr_collapse_R", 0) or 0)
    head_mult = float(cfg.get("lr_head_mult", 1.0))
    grad_mult = float(cfg.get("lr_grad_mult", 1.0))
    global_mult = float(cfg.get("lr_global_mult", 1.0))
    f64 = bool(cfg.get("lr_float64", False))
    trace_every = int(cfg.get("lr_trace_every", 0) or 0)
    classical = bool(cfg.get("lr_classical"))
    R_built = int(cfg.get("lr_classical_R", 1))

    def post_build(dqn):
        # The library draws random exploration actions from an action-space generator that no seed reaches.
        dqn.envs.single_action_space.seed(int(dqn.seed))
        q = dqn.q_network
        if f64:
            to_float64_(q)
        R_now = R_built
        if collapse:
            if classical:
                if collapse != R_built:
                    raise RuntimeError(f"lr_collapse_R={collapse} but lr_classical_R={R_built}")
                collapse_blocks_(q, collapse)
                R_now = 1
            else:
                got = collapse_head_(q)
                if got != collapse:
                    raise RuntimeError(f"lr_collapse_R={collapse} but the model has OutputReuse({got})")
        # The target network is rebuilt from the online one: it was built before the changes above.
        dqn.target_network = copy.deepcopy(q)
        if classical:
            if head_mult != 1.0:
                hw = head_linear(q).weight
                dqn.optimizer = _split_adam(q, hw, dqn.learning_rate, dqn.learning_rate * head_mult, dqn.learning_rate)
            elif f64 or collapse:
                dqn.optimizer = torch.optim.Adam(q.parameters(), lr=dqn.learning_rate)
            return BlocksTraceHook(dqn, R_now, trace_every)
        if head_mult != 1.0 or global_mult != 1.0:
            dqn.optimizer = build_adam(q, dqn.learning_rate, head_mult, global_mult)
        elif f64 or collapse:
            dqn.optimizer = torch.optim.Adam(q.parameters(), lr=dqn.learning_rate)
        if grad_mult != 1.0:
            scale_grad_(head_weight(q), grad_mult)
        return TraceHook(dqn, trace_every)

    return post_build


def v_layers(agent) -> List[nn.Linear]:
    """The linear layers that read the (replicated) input directly: first layer of the actor and of the critic."""
    return [agent.actor[0], agent.critic[0]]


def reuse_of(agent) -> int:
    return int(agent.input_reuse.n_repeats) if getattr(agent, "input_reuse", None) is not None else 1


def collapse_agent_(agent) -> int:
    """Drop OutputReuse(R) and give the actor and the critic a first layer with V^0 = sum_r W_r^0 and the same bias; returns R."""
    R = reuse_of(agent)
    if R == 1:
        return 1
    for seq in (agent.actor, agent.critic):
        old = seq[0]
        out, inn = old.weight.shape
        f = inn // R
        with torch.random.fork_rng(devices=[]):
            new = nn.Linear(f, out).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(old.weight.reshape(out, R, f).sum(dim=1))
            new.bias.copy_(old.bias)
        seq[0] = new
    agent.input_reuse = None
    return R


class LRPPO(PPO):
    """The PPO of the reference study with a learning rate per group, optional or modified gradient clipping and greedy evaluation.

    ``train`` is the library loop with three changes: the annealing multiplies every parameter group, the gradient is clipped by
    ``_norm_and_clip`` (clip_mode "torch" is the library rule, "none" disables it, "modified" uses the norm that the reuse-R network
    would have) and the deterministic policy is evaluated every ``eval_every`` steps. With the defaults the returns are those of ``PPO.train``.
    """

    def __init__(self, *args, head_lr_mult: float = 1.0, collapse: bool = False, clip_mode: str = "torch",
                 clip_R: int = 1, eval_every: int = 0, eval_episodes: int = 20, eval_env_id: str = "CartPole-v1",
                 float64: bool = False, **kwargs):
        if clip_mode not in ("torch", "none", "modified"):
            raise ValueError(f"clip_mode must be torch|none|modified, got {clip_mode!r}")
        super().__init__(*args, **kwargs)
        self.head_lr_mult = float(head_lr_mult)
        self.clip_mode = clip_mode
        self.clip_R = int(clip_R)
        self.eval_every, self.eval_episodes, self.eval_env_id = int(eval_every), int(eval_episodes), eval_env_id
        if collapse:
            got = collapse_agent_(self.agent)
            if clip_mode == "modified" and got != self.clip_R:
                raise ValueError(f"clip_R={self.clip_R} but the agent had OutputReuse({got})")
        if float64:
            self.agent.double()
            for m in (self.agent.actor, self.agent.critic, self.agent.input_reuse):
                if m is not None:
                    m.register_forward_pre_hook(_cast_inputs_float64)
        self._v_weights = [m.weight for m in v_layers(self.agent)]
        if self.head_lr_mult != 1.0 or collapse or float64:
            v_ids = {id(w) for w in self._v_weights}
            rest = [p for p in self.agent.parameters() if id(p) not in v_ids]
            self.optimizer = torch.optim.Adam(
                [{"params": self._v_weights, "lr": self.learning_rate * self.head_lr_mult},
                 {"params": rest, "lr": self.learning_rate}], lr=self.learning_rate, eps=1e-5)
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.clip_log: List[tuple] = []
        self.eval_rows: List[tuple] = []
        self.iter_hook = None

    def _norm_and_clip(self) -> None:
        params = [p for p in self.agent.parameters() if p.grad is not None]
        if self.clip_mode == "torch":
            norm = float(nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm))
        else:
            v_ids = {id(w) for w in self._v_weights}
            w_ = self.clip_R if self.clip_mode == "modified" else 1
            sq = sum((w_ if id(p) in v_ids else 1) * float(torch.linalg.vector_norm(p.grad)) ** 2 for p in params)
            norm = math.sqrt(sq)
            if self.clip_mode == "modified":
                coef = min(self.max_grad_norm / (norm + 1e-6), 1.0)
                for p in params:
                    p.grad.mul_(coef)
        factor = min(1.0, self.max_grad_norm / (norm + 1e-6))
        self.clip_log.append((norm, factor, norm > self.max_grad_norm))


    def _greedy_eval(self) -> float:
        env = gym.make(self.eval_env_id)
        rets = []
        sel = list(REFERENCE_COMPONENTS)
        try:
            with torch.no_grad():
                for k in range(self.eval_episodes):
                    obs, _ = env.reset(seed=self.seed + 10_000 + k)
                    done, total = False, 0.0
                    while not done:
                        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)[sel]).unsqueeze(0)
                        if self.agent.input_reuse is not None:
                            x = self.agent.input_reuse(x)
                        a = int(torch.argmax(self.agent.actor(x), dim=1).item())
                        obs, r, term, trunc, _ = env.step(a)
                        total += float(r)
                        done = bool(term) or bool(trunc)
                    rets.append(total)
        finally:
            env.close()
        return float(np.mean(rets))


    def train(self, total_timesteps: int = 100_000, progress_bar: bool = False):
        """Upstream ``PPO.train`` with the four additions described in the module docstring (marked )."""
        next_obs_np, _ = self.envs.reset(seed=self.seed)
        batch = next_obs_np.reshape(self.num_envs, -1)
        next_obs = torch.tensor(batch, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        self.num_iterations = math.ceil(total_timesteps / self.batch_size)
        pbar = tqdm(total=total_timesteps, desc="Training", unit="step") if progress_bar else None
        next_eval = self.eval_every if self.eval_every else None
        v_loss = pg_loss = None

        for iteration in range(1, self.num_iterations + 1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                for g, base in zip(self.optimizer.param_groups, self._base_lrs):
                    g["lr"] = frac * base

            for step in range(self.num_steps):
                self.global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.actions[step] = action
                self.logprobs[step] = logprob
                self.values[step] = value.flatten()

                next_obs_np, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminated, truncated)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)

                batch = next_obs_np.reshape(self.num_envs, -1)
                next_obs = torch.tensor(batch, dtype=torch.float32, device=self.device)
                next_done = torch.Tensor(next_done).to(self.device)

                if pbar is not None:
                    pbar.update(1)

                if "episode" in infos and "_episode" in infos:
                    for i, done in enumerate(infos["_episode"]):
                        if done:
                            ep_return = float(infos["episode"]["r"][i])
                            ep_length = int(infos["episode"]["l"][i])
                            self.writer.add_scalar("charts/episodic_return", ep_return, self.global_step)
                            self.writer.add_scalar("charts/episodic_length", ep_length, self.global_step)
                            self.episode_counter += 1
                            self.csv_writer.writerow([self.episode_counter, ep_return, self.global_step])
                            self.csv_file.flush()

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            b_obs = self.obs.reshape(self.batch_size, -1)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    self._norm_and_clip()
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)

            if self.iter_hook is not None:
                self.iter_hook(self)
            if next_eval is not None:
                while next_eval <= self.global_step and next_eval <= total_timesteps:
                    self.eval_rows.append((next_eval, self._greedy_eval()))
                    next_eval += self.eval_every

        if pbar is not None:
            pbar.close()
        self.envs.close()
        self.writer.close()
        self.csv_file.close()
        print("Training finished!")

    def write_outputs(self, run_name: str, outdir: pathlib.Path) -> Dict[str, Optional[str]]:
        outdir = pathlib.Path(outdir)
        paths: Dict[str, Optional[str]] = {"eval_csv": None}
        runs = outdir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        if self.eval_rows:
            p = runs / f"{run_name}_eval.csv"
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["global_step", "greedy_return"])
                w.writerows(self.eval_rows)
            paths["eval_csv"] = str(p)
        return paths


PPO_ENV = "CartPole-v1"
PPO_STEPS = 100_000
EVAL_EVERY, EVAL_EPISODES = 10_000, 20

# Hyperparameters of the reference study; none is tuned here.
REFERENCE_PPO_CONFIG: Dict[str, Any] = {
    "learning_rate": 2.5e-4, "num_envs": 1, "num_steps": 512, "anneal_lr": True, "gamma": 0.99, "gae_lambda": 0.95,
    "num_minibatches": 16, "update_epochs": 4, "norm_adv": True, "clip_coef": 0.2, "clip_vloss": True,
    "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "target_kl": None,
}

RS_REUSE = (1, 4, 8, 16, 32)
RS_LRX = (4, 8, 16, 32)
RS_MODIFIED = (8, 32)


@dataclass(frozen=True)
class PPOArm:
    name: str
    R: int
    collapse: bool
    clip_mode: str


def ppo_arms() -> List[PPOArm]:
    out = [PPOArm(f"ppo_reference_reuse{R}_clip", R, False, "torch") for R in RS_REUSE]
    out += [PPOArm(f"ppo_reference_reuse{R}_noclip", R, False, "none") for R in RS_REUSE]
    out += [PPOArm(f"ppo_reference_lr_x{R}_clip", R, True, "torch") for R in RS_LRX]
    out += [PPOArm(f"ppo_reference_lr_x{R}_noclip", R, True, "none") for R in RS_LRX]
    out += [PPOArm(f"ppo_reference_lr_x{R}_modified_clip", R, True, "modified") for R in RS_MODIFIED]
    return out


PPO_ARMS = {a.name: a for a in ppo_arms()}


@dataclass
class PPOSpec:
    arm: str
    seed: int
    total_timesteps: int = PPO_STEPS
    tag: str = ""
    ppo_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(REFERENCE_PPO_CONFIG))

    @property
    def run_name(self) -> str:
        base = f"{self.arm}__s{self.seed}"
        return f"{base}__{self.tag}" if self.tag else base


def build_ppo(arm: PPOArm, seed: int, run_name: str, eval_every: int = EVAL_EVERY, eval_episodes: int = EVAL_EPISODES,
              ppo_kwargs: Dict[str, Any] = None) -> "LRPPO":
    kw = dict(REFERENCE_PPO_CONFIG if ppo_kwargs is None else ppo_kwargs)
    return LRPPO(env=make_selected_env(PPO_ENV), seed=seed, agent_type="mlp", agent_config={"n_repeats": arm.R, "net_arch": []},
                 run_name=run_name, head_lr_mult=float(arm.R) if arm.collapse else 1.0, collapse=arm.collapse,
                 clip_mode=arm.clip_mode, clip_R=arm.R, eval_every=eval_every, eval_episodes=eval_episodes, **kw)


def run_ppo_cell(spec: PPOSpec, outdir: pathlib.Path, eval_episodes: int = EVAL_EPISODES) -> Dict[str, Any]:
    """Train one PPO cell; the episode and greedy-evaluation CSVs go to ``outdir/runs``."""
    outdir = pathlib.Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    t0 = time.time()
    try:
        os.chdir(outdir)
        ppo = build_ppo(PPO_ARMS[spec.arm], spec.seed, spec.run_name, eval_episodes=eval_episodes, ppo_kwargs=spec.ppo_kwargs)
        ppo.train(total_timesteps=spec.total_timesteps)
        ppo.write_outputs(spec.run_name, outdir)
    finally:
        os.chdir(cwd)
    return {"run_name": spec.run_name, "wall_seconds": round(time.time() - t0, 1)}
