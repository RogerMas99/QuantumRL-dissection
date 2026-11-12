"""PPO of the reference study with a per-group learning rate, optional / modified gradient clipping
and a clipping diagnostic (docs/design/EXPERIMENT-13-OR-VS-LR.md).

``LRPPO`` subclasses the vendored ``simplyqrl.ppo.PPO`` (untouched). ``train`` is the upstream loop line for line with
FOUR additions, none of which changes the behaviour when its option is left at the default (tested: with the defaults
the episode returns are bit-for-bit those of ``PPO.train``):

  1. learning rate by groups: the weight matrices that read the replicated observation (``actor[0].weight`` and
     ``critic[0].weight``) can carry ``head_lr_mult`` x the base learning rate; the annealing multiplies EVERY group
     (upstream only sets ``param_groups[0]``, which is the only group it has);
  2. ``clip_mode``: "torch" (upstream: ``clip_grad_norm_`` over all parameters with ``max_grad_norm``), "none"
     (no clipping) or "modified" (the same clipping, but the norm is sqrt(R * ||g_V||^2 + ||g_rest||^2): the norm the
     reuse-R network would have; V = the weights of 1.);
  3. clipping diagnostic: at every update the norm of the gradient BEFORE clipping, the factor
     min(1, max_grad_norm / (norm + 1e-6)) that the rule gives and whether the clipping fired (for "none" it is what
     would have happened with the unmodified rule);
  4. greedy evaluation (deterministic argmax of the actor) every ``eval_every`` steps with ``eval_episodes`` episodes on a
     separate environment, plus diagnostics every 1000 steps: ||V_eff||_F of the actor and of the critic, mean |value|
     over the rollout and the last value / policy loss.

The agent is the reference study's classical control: the observation reduced to indices [1, 2, 3] (cart position
discarded) and replicated R times (``OutputReuse(R)``), then one linear layer for the actor and one for the critic
(``net_arch = []``). The reduction is done by an observation wrapper, so the agent receives 3 features; this is the
same function as upstream's ``SelectiveOutputReuse`` on the 4-dimensional observation (tested: identical initial
weights under one seed and identical outputs).
"""
from __future__ import annotations

import csv
import math
import pathlib
import time
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from simplyqrl.ppo import PPO
from tqdm import tqdm

from quantumrl_dissection.environments import REFERENCE_COMPONENTS as REUSE_INDICES, SelectObs, make_selected_env  # noqa: E402,F401


def _cast64(module, args):
    return tuple(a.to(torch.float64) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)


def effective_flat(ppo) -> np.ndarray:
    """V_eff and bias of the actor and of the critic, flattened (float64): the quantity the equivalence preserves."""
    out = []
    R = reuse_of(ppo.agent)
    for seq in (ppo.agent.actor, ppo.agent.critic):
        w, b = seq[0].weight.detach().double(), seq[0].bias.detach().double()
        o, i = w.shape
        out += [(w.reshape(o, R, i // R).sum(dim=1) if R > 1 else w).reshape(-1), b.reshape(-1)]
    return torch.cat(out).cpu().numpy()


def v_layers(agent) -> List[nn.Linear]:
    """The linear layers that read the (replicated) input directly: first layer of the actor and of the critic."""
    return [agent.actor[0], agent.critic[0]]


def reuse_of(agent) -> int:
    return int(agent.input_reuse.n_repeats) if getattr(agent, "input_reuse", None) is not None else 1


def collapse_agent_(agent) -> int:
    """In place: drop OutputReuse(R) and give the actor and the critic a first layer with V^0 = sum_r W_r^0 and the same
    bias. Keeps torch's RNG stream where it was. Returns R."""
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
    def __init__(self, *args, head_lr_mult: float = 1.0, collapse: bool = False, clip_mode: str = "torch",
                 clip_R: int = 1, eval_every: int = 0, eval_episodes: int = 20, eval_env_id: str = "CartPole-v1",
                 diag_every: int = 0, float64: bool = False, **kwargs):
        if clip_mode not in ("torch", "none", "modified"):
            raise ValueError(f"clip_mode must be torch|none|modified, got {clip_mode!r}")
        super().__init__(*args, **kwargs)
        self.head_lr_mult = float(head_lr_mult)
        self.clip_mode = clip_mode
        self.clip_R = int(clip_R)
        self.eval_every, self.eval_episodes, self.eval_env_id = int(eval_every), int(eval_episodes), eval_env_id
        self.diag_every = int(diag_every)
        if collapse:
            got = collapse_agent_(self.agent)
            if clip_mode == "modified" and got != self.clip_R:
                raise ValueError(f"clip_R={self.clip_R} but the agent had OutputReuse({got})")
        if float64:     # trace tests: arithmetic of the agent in float64 (call torch.set_default_dtype first for buffers)
            self.agent.double()
            for m in (self.agent.actor, self.agent.critic, self.agent.input_reuse):
                if m is not None:
                    m.register_forward_pre_hook(_cast64)
        self._v_weights = [m.weight for m in v_layers(self.agent)]
        if self.head_lr_mult != 1.0 or collapse or float64:
            v_ids = {id(w) for w in self._v_weights}
            rest = [p for p in self.agent.parameters() if id(p) not in v_ids]
            self.optimizer = torch.optim.Adam(          # same arguments as upstream (eps = 1e-5), two groups
                [{"params": self._v_weights, "lr": self.learning_rate * self.head_lr_mult},
                 {"params": rest, "lr": self.learning_rate}], lr=self.learning_rate, eps=1e-5)
        self._base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.clip_log: List[tuple] = []          # (norm_before_clipping, factor, fired)
        self.eval_rows: List[tuple] = []
        self.diag_rows: List[tuple] = []
        self.iter_hook = None                    # optional callable(ppo) run after every iteration (trace tests)

    # ----------------------------------------------------------------------------------------------------------
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

    def clip_summary(self) -> Dict[str, Any]:
        if not self.clip_log:
            return {"n_updates": 0}
        a = np.asarray(self.clip_log, dtype=float)
        return {"n_updates": int(len(a)), "clip_fraction": float(a[:, 2].mean()),
                "median_factor": float(np.median(a[:, 1])), "median_norm": float(np.median(a[:, 0])),
                "mean_norm": float(a[:, 0].mean()), "p90_norm": float(np.percentile(a[:, 0], 90)),
                "mode": self.clip_mode, "max_grad_norm": self.max_grad_norm}

    def _greedy_eval(self) -> float:
        env = gym.make(self.eval_env_id)
        rets = []
        sel = list(REUSE_INDICES)
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

    def _v_eff(self, seq) -> float:
        R = reuse_of(self.agent)
        w = seq[0].weight.detach().double()
        out, inn = w.shape
        return float(torch.linalg.norm(w.reshape(out, R, inn // R).sum(dim=1) if R > 1 else w))

    # ----------------------------------------------------------------------------------------------------------
    def train(self, total_timesteps: int = 100_000, progress_bar: bool = False):
        """Upstream ``PPO.train`` with the four additions described in the module docstring (marked )."""
        next_obs_np, _ = self.envs.reset(seed=self.seed)
        batch = next_obs_np.reshape(self.num_envs, -1)
        next_obs = torch.tensor(batch, dtype=torch.float32, device=self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        start_time = time.time()

        self.num_iterations = math.ceil(total_timesteps / self.batch_size)
        pbar = tqdm(total=total_timesteps, desc="Training", unit="step") if progress_bar else None
        next_eval = self.eval_every if self.eval_every else None
        next_diag = self.diag_every if self.diag_every else None
        v_loss = pg_loss = None

        for iteration in range(1, self.num_iterations + 1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                for g, base in zip(self.optimizer.param_groups, self._base_lrs):        # every group
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
                    self._norm_and_clip()                                                    # replaces clip_grad_norm_
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
            # diagnostics and greedy evaluation; neither draws from any RNG stream
            if next_diag is not None and self.global_step >= next_diag:
                self.diag_rows.append((self.global_step, self._v_eff(self.agent.actor), self._v_eff(self.agent.critic),
                                       float(self.values.abs().mean()), float(v_loss.detach()), float(pg_loss.detach())))
                while next_diag <= self.global_step:
                    next_diag += self.diag_every
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

    # ----------------------------------------------------------------------------------------------------------
    def write_outputs(self, run_name: str, outdir: pathlib.Path) -> Dict[str, Optional[str]]:
        outdir = pathlib.Path(outdir)
        paths: Dict[str, Optional[str]] = {"eval_csv": None, "diag_csv": None, "clip_npy": None}
        runs = outdir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        if self.eval_rows:
            p = runs / f"{run_name}_eval.csv"
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["global_step", "greedy_return"])
                w.writerows(self.eval_rows)
            paths["eval_csv"] = str(p)
        if self.diag_rows:
            p = runs / f"{run_name}_diag.csv"
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["step", "v_eff_fro_actor", "v_eff_fro_critic", "mean_abs_value", "value_loss", "policy_loss"])
                w.writerows(self.diag_rows)
            paths["diag_csv"] = str(p)
        if self.clip_log:
            p = runs / f"{run_name}_clip.npy"
            np.save(p, np.asarray(self.clip_log, dtype=np.float32))
            paths["clip_npy"] = str(p)
        return paths


# ---------------------------------------------------------------------------------------------------------------------
# Arms and runs of the learning-rate experiment under PPO
#
# Agent: the reference study's classical control (observation reduced to the components [1, 2, 3], replicated R times,
# then one linear layer for the actor and one for the critic; ``net_arch = []``), trained with ``LRPPO``. The
# hyperparameters are those of the reference study's repository (``REFERENCE_PPO_CONFIG``); none is tuned.
#
#   ppo_reference_reuse{R}_{clip,noclip}      reuse R, with / without clipping of the gradient    R = 1, 4, 8, 16, 32
#   ppo_reference_lr_x{R}_{clip,noclip}       no reuse, V^0 = sum_r W_r^0, lr of the weights that read the replicated input
#                                             x R, with / without clipping                         R = 4, 8, 16, 32
#   ppo_reference_lr_x{R}_modified_clip       the same with the clipping norm sqrt(R ||g_V||^2 + ||g_rest||^2)   R = 8, 32
# ---------------------------------------------------------------------------------------------------------------------
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402

PPO_ENV = "CartPole-v1"
PPO_STEPS = 100_000
EVAL_EVERY, EVAL_EPISODES, DIAG_EVERY = 10_000, 20, 1_000

# Verbatim from the reference study's repository (github.com/javier-lazaro/qrl-dissection, src/experiments/*.py).
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
    collapse: bool          # True: the learning-rate arm (reuse-R initialisation, head collapsed)
    clip_mode: str          # torch | none | modified


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
              diag_every: int = DIAG_EVERY, ppo_kwargs: Dict[str, Any] = None) -> "LRPPO":
    kw = dict(REFERENCE_PPO_CONFIG if ppo_kwargs is None else ppo_kwargs)
    return LRPPO(env=make_selected_env(PPO_ENV), seed=seed, agent_type="mlp",
                 agent_config={"n_repeats": arm.R, "net_arch": []}, run_name=run_name,
                 head_lr_mult=float(arm.R) if arm.collapse else 1.0, collapse=arm.collapse, clip_mode=arm.clip_mode,
                 clip_R=arm.R, eval_every=eval_every, eval_episodes=eval_episodes, diag_every=diag_every, **kw)


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=pathlib.Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return ""


def run_ppo_cell(spec: PPOSpec, outdir: pathlib.Path, force: bool = False,
                 eval_episodes: int = EVAL_EPISODES) -> Dict[str, Any]:
    """Run one cell, or return its manifest if it exists for the same number of steps."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mpath = outdir / f"{spec.run_name}.manifest.json"
    if mpath.exists() and not force:
        stored = json.loads(mpath.read_text())
        if stored.get("spec", {}).get("total_timesteps") == spec.total_timesteps:
            return stored
    arm = PPO_ARMS[spec.arm]
    cwd = os.getcwd()
    t0 = time.time()
    try:
        os.chdir(outdir)
        ppo = build_ppo(arm, spec.seed, spec.run_name, eval_episodes=eval_episodes, ppo_kwargs=spec.ppo_kwargs)
        ppo.train(total_timesteps=spec.total_timesteps)
        paths = ppo.write_outputs(spec.run_name, outdir)
    finally:
        os.chdir(cwd)
    manifest = {
        "spec": asdict(spec),
        "run_name": spec.run_name,
        "outcome": {"run_name": spec.run_name, "seed": spec.seed, "total_timesteps": spec.total_timesteps,
                    "wall_seconds": round(time.time() - t0, 1), "probe": {},
                    "episodes_csv": str(outdir / "runs" / f"{spec.run_name}.csv"),
                    "eval_csv": paths["eval_csv"], "diag_csv": paths["diag_csv"], "clip_npy": paths["clip_npy"],
                    "clip": ppo.clip_summary()},
        "agent_type": "mlp",
        "agent_config": {"n_repeats": arm.R, "net_arch": [], "obs_indices": [1, 2, 3], "collapse": arm.collapse,
                         "head_lr_mult": float(arm.R) if arm.collapse else 1.0, "clip_mode": arm.clip_mode},
        "env_id": PPO_ENV,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
    }
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
