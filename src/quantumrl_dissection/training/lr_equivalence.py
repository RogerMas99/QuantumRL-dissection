"""Output Reuse against the learning rate of the head (docs/design/EXPERIMENT-13-OR-VS-LR.md).

Proposition under test (thesis section 3.4.2). With z the features that reach the linear head and
q = [W_1|...|W_R][z; ...; z] + b (what ``OutputReuse(R)`` + the first head ``Linear`` compute):

  (i)  the network depends on the copies only through V = sum_r W_r;
  (ii) with SGD, momentum or Adam, no global gradient-norm clipping and no weight decay, training is the same as
       for the network without reuse (head matrix V) if the learning rate of V is multiplied by R and
       V^0 = sum_r W_r^0. The learning rate of the bias does not change.

Everything here works on a constructed ``DQN`` (``simplyqrl.dqn``) AFTER ``__init__`` and BEFORE ``train``, through
the ``post_build`` hook of ``dqn.safe.SafeDQN``; the vendored library is not touched.

Config keys (read from the arm's ``agent_config``; the agent classes ignore keys they do not know):

  lr_diag          True  -> attach the diagnostics of ``LearningRateHook`` (set by the stage of the learning-rate experiment for every arm)
  lr_collapse_R    int   -> replace the head of the reuse-R model by one with V^0 = sum_r W_r^0 (no OutputReuse)
  lr_head_mult     float -> learning rate of ``head.weight`` (only that tensor) times this factor
  lr_grad_mult     float -> gradient of ``head.weight`` times this factor, learning rate untouched
  lr_global_mult   float -> learning rate of ALL parameters times this factor
  lr_float64       True  -> network arithmetic in float64 (inputs are cast on entry); used by the trace tests
  lr_trace_every   int   -> keep the effective parameters and the actions every that many steps (trace tests)

Every arm with an lr_* key also seeds upstream's exploration sampler with the run's seed (see post_build).

"head.weight" is the weight of the first ``nn.Linear`` of ``agent.network`` after the front end (circuit or
classical extractor): the matrix that reads the replicated features. Its bias, the second head layer and
the front end keep the base learning rate.
"""
from __future__ import annotations

import copy
import csv
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

KEYS = ("lr_diag", "lr_classical", "lr_collapse_R", "lr_head_mult", "lr_grad_mult", "lr_global_mult", "lr_float64")


def has_lr(cfg: Optional[Dict[str, Any]]) -> bool:
    return bool(cfg) and any(k in cfg for k in KEYS)


# ---------------------------------------------------------------------------------------------------------------
# Locating and collapsing the head
# ---------------------------------------------------------------------------------------------------------------
def head_index(net: nn.Sequential) -> int:
    """Index (in ``agent.network``) of the first top-level ``nn.Linear``: the layer that reads the replicated
    features. The front end (a PennyLane/statevector layer or the twin's ``nn.Sequential`` extractor) is not a
    top-level Linear, so it is skipped."""
    for i, m in enumerate(net):
        if isinstance(m, nn.Linear):
            return i
    raise RuntimeError("no top-level nn.Linear in agent.network")


def reuse_factor(net: nn.Sequential) -> int:
    """R of the OutputReuse module in ``net`` (1 if there is none)."""
    for m in net:
        if type(m).__name__ == "OutputReuse":
            return int(m.n_repeats)
    return 1


def head_weight(agent: nn.Module) -> nn.Parameter:
    return agent.network[head_index(agent.network)].weight


def effective_v(agent: nn.Module) -> torch.Tensor:
    """V_eff = sum_r W_r (shape out x f). Without reuse it is the head weight itself."""
    net = agent.network
    w = net[head_index(net)].weight
    R = reuse_factor(net)
    out, inn = w.shape
    return w.reshape(out, R, inn // R).sum(dim=1) if R > 1 else w


def collapse_head_(agent: nn.Module) -> int:
    """In place: replace OutputReuse(R) + Linear(f*R -> out) by Linear(f -> out) with V^0 = sum_r W_r^0 and the same
    bias. Every other module (front end, second head layer) is kept as the same object, with its weights.
    Returns R (1 means there was nothing to collapse)."""
    net = agent.network
    R = reuse_factor(net)
    if R == 1:
        return 1
    i = head_index(net)
    old: nn.Linear = net[i]
    out, inn = old.weight.shape
    f = inn // R
    # nn.Linear.__init__ draws from torch's global RNG; keep the stream where it was (matters for PPO, whose rollouts
    # sample actions from it, so the collapsed and the reuse arm must see the same stream).
    with torch.random.fork_rng(devices=[]):
        new = nn.Linear(f, out, bias=old.bias is not None).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight.reshape(out, R, f).sum(dim=1))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    layers = []
    for j, m in enumerate(net):
        if type(m).__name__ == "OutputReuse":
            continue
        layers.append(new if j == i else m)
    agent.network = nn.Sequential(*layers)
    if hasattr(agent, "reuse_repetitions"):
        agent.reuse_repetitions = 1
    return R


# ---------------------------------------------------------------------------------------------------------------
# Optimiser: parameter groups
# ---------------------------------------------------------------------------------------------------------------
def build_adam(agent: nn.Module, base_lr: float, head_mult: float = 1.0, global_mult: float = 1.0) -> torch.optim.Adam:
    """Adam with the same default arguments as ``simplyqrl.dqn`` (only ``lr``), split in two groups: head.weight
    (lr = base * global * head) and everything else (lr = base * global). No weight decay, no scheduler."""
    hw = head_weight(agent)
    rest = [p for p in agent.parameters() if p is not hw]
    groups = [{"params": [hw], "lr": base_lr * global_mult * head_mult},
              {"params": rest, "lr": base_lr * global_mult}]
    return torch.optim.Adam(groups, lr=base_lr)


def scale_grad_(param: nn.Parameter, factor: float):
    """Multiply the gradient of ``param`` by ``factor`` as it is produced (after backward, before step)."""
    return param.register_hook(lambda g: g * factor)


def _cast_inputs_float64(module, args):
    return tuple(a.to(torch.float64) if torch.is_tensor(a) and a.is_floating_point() else a for a in args)


def to_float64_(agent: nn.Module) -> nn.Module:
    agent.double()
    agent.register_forward_pre_hook(_cast_inputs_float64)
    return agent


# ---------------------------------------------------------------------------------------------------------------
# Diagnostics, attached through the probe: every 1000 steps ||V_eff||_F, mean |Q| on the last states of the buffer
# and the last logged TD loss; optionally a dense trace of the effective parameters (float64 trace tests).
# Nothing here draws from any RNG, so attaching it does not change the trajectory.
# ---------------------------------------------------------------------------------------------------------------
class _TeeWriter:
    """Delegates to the real SummaryWriter and remembers the last TD loss that DQN logged (every 100 steps)."""

    def __init__(self, inner, store: "LearningRateHook"):
        self._inner, self._store = inner, store

    def add_scalar(self, tag, value, *args, **kwargs):
        if tag == "losses/td_loss":
            self._store.last_td = float(value.detach()) if torch.is_tensor(value) else float(value)
        return self._inner.add_scalar(tag, value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class LearningRateHook:
    COLUMNS = ["step", "v_eff_fro", "mean_abs_q", "td_loss"]

    def __init__(self, dqn, every: int = 1000, n_states: int = 128, trace_every: int = 0):
        self.dqn, self.every, self.n_states, self.trace_every = dqn, every, n_states, trace_every
        self.rows: List[Tuple] = []
        self.last_td = float("nan")
        self.actions: List[int] = []          # actions taken, for the trace comparison
        self.trace_steps: List[int] = []
        self.trace_params: List[np.ndarray] = []
        dqn.writer = _TeeWriter(dqn.writer, self)
        if trace_every:
            orig = dqn.envs.step

            def _step(actions, _orig=orig):
                self.actions.append(int(np.asarray(actions).reshape(-1)[0]))
                return _orig(actions)

            dqn.envs.step = _step             # SafeDQN's probe wraps this again and restores its own original

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
        if n % self.every:
            return
        rb, q = self.dqn.rb, self.dqn.q_network
        size = rb.max_size if rb.full else rb.pos
        k = min(self.n_states, size)
        idx = (np.arange(rb.pos - k, rb.pos) % rb.max_size) if k else np.arange(0)
        with torch.no_grad():
            vfro = float(torch.linalg.norm(effective_v(q).double()))
            if k:
                obs = torch.as_tensor(rb.obs_buf[idx], dtype=torch.get_default_dtype())
                mq = float(q(obs).abs().double().mean())
            else:
                mq = float("nan")
        self.rows.append((n, vfro, mq, self.last_td))

    def write(self, path) -> str:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.COLUMNS)
            w.writerows(self.rows)
        return str(path)


def make_post_build(cfg: Optional[Dict[str, Any]]):
    """Return ``post_build(dqn) -> LearningRateHook`` for an arm config, or None when the arm has no lr_* key (then the
    pipeline is the unchanged one)."""
    if not has_lr(cfg):
        return None
    if cfg.get("lr_classical"):                  # blocks E, F, G (core/lr_efg.py)
        from quantumrl_dissection.training import lr_classical as lr_efg
        return lr_efg.make_post_build(cfg)
    collapse = int(cfg.get("lr_collapse_R", 0) or 0)
    head_mult = float(cfg.get("lr_head_mult", 1.0))
    grad_mult = float(cfg.get("lr_grad_mult", 1.0))
    global_mult = float(cfg.get("lr_global_mult", 1.0))
    f64 = bool(cfg.get("lr_float64", False))
    trace_every = int(cfg.get("lr_trace_every", 0) or 0)

    def post_build(dqn) -> LearningRateHook:
        # Upstream's epsilon-greedy draws its random actions from an action-space generator that no seed reaches
        # (docs/adjustments.md, "action-space sampling"). Seeding it with the run's seed leaves the exploration
        # distribution untouched and makes a cell reproducible, which the same-seed comparisons of the learning-rate experiment need.
        dqn.envs.single_action_space.seed(int(dqn.seed))
        q = dqn.q_network
        if f64:
            to_float64_(q)
        if collapse:
            got = collapse_head_(q)
            if got != collapse:
                raise RuntimeError(f"lr_collapse_R={collapse} but the model has OutputReuse({got})")
        # The target network must be the same object kind as the online one, with the same weights: it was built
        # (and, in float64, converted) before the surgery, so it is rebuilt from the online network.
        dqn.target_network = copy.deepcopy(q)
        if head_mult != 1.0 or global_mult != 1.0:
            dqn.optimizer = build_adam(q, dqn.learning_rate, head_mult, global_mult)
        elif f64 or collapse:
            dqn.optimizer = torch.optim.Adam(q.parameters(), lr=dqn.learning_rate)   # parameters were replaced
        if grad_mult != 1.0:
            scale_grad_(head_weight(q), grad_mult)
        return LearningRateHook(dqn, trace_every=trace_every)

    return post_build


# ---------------------------------------------------------------------------------------------------------------
# Constants of the arm registry (arms.py) and the names of the arms
# ---------------------------------------------------------------------------------------------------------------
RS_EQUIV = (1, 4, 8, 16, 32)
RS_COLLAPSED = (4, 8, 16, 32)
RS_GRAD = (8, 32)
KS_HEAD = (2, 4, 8, 16, 32, 64, 128)
KS_GLOBAL = (0.5, 2, 4, 8)
PREFIX_TWIN, PREFIX_HYBRID = "classical_twin", "sim_statevector_hybrid"


def kname(k) -> str:
    return str(k).replace(".", "p")


def arm_names(prefix: str) -> Dict[str, List[str]]:
    """Names of the arms of the learning-rate experiment for one agent (see ``arms.py``)."""
    reuse = [prefix if R == 1 else f"{prefix}_reuse{R}" for R in RS_EQUIV]
    return {
        "A_reuse": reuse,
        "A_collapsed": [f"{prefix}_collapsed{R}" for R in RS_COLLAPSED],
        "A_grad": [f"{prefix}_grad_x{R}" for R in RS_GRAD],
        "B_headlr": [f"{prefix}_head_lr_x{kname(k)}" for k in KS_HEAD],
        "C_global": [f"{prefix}_all_lr_x{kname(k)}" for k in KS_GLOBAL],
    }
