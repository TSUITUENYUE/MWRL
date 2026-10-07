"""GFlowNet baseline on the MaxSAT add-a-variable MDP (trajectory-balance objective).

The learned diversity-seeking baseline: it samples terminal witnesses with probability proportional
to a per-object reward R, so it spreads across many witnesses rather than collapsing to one like a
scalar RL optimum. Two reward variants match the paper's rows:
  * success  -- R = [S is a witness]; uniform over all satisfying sets.
  * up-set   -- R = nu(up S) = p^{|S|} for a witness; the coverage measure as a per-object reward.
Because R is a per-OBJECT value (not a leave-one-out marginal), proportional sampling puts most mass
on the exponentially many non-minimal supersets, so raw proposals are rarely minimal -- the contrast
with MWRL's marginal-coverage credit. Evaluated with the same antichain scorer (``evaluate``).
``PenalizedGFlowNet`` generalizes both variants to R = exp(-lam |S|), the size-penalty sweep.

Trajectory balance with a uniform backward policy: on the add-only DAG a size-k terminal has k!
add-orderings, so log P_B(tau) = -log(k!), and the TB residual is
    log Z + sum_t log P_F(a_t|s_t) - log R(x) + log(k!).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from mwrl.modules import Actor
from mwrl.runner import Rollout
from torch import nn

from mwrl_maxsat.env import MaxSatEnv


class GFlowNet:
    def __init__(self, env: MaxSatEnv, *, reward: str = "upset", p: float = 0.7, hidden_dim: int = 128,
                 lr: float = 3e-3, device: str = "cpu", seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.env = env
        self.reward = reward                 # "success" | "upset"
        self.p = p
        self.device = torch.device(device)
        self.pf = Actor(env.num_obs, env.num_actions, hidden_dim=hidden_dim).to(self.device)
        self.logZ = nn.Parameter(torch.zeros((), device=self.device))
        self.opt = torch.optim.Adam(list(self.pf.parameters()) + [self.logZ], lr=lr)
        self.rng = np.random.default_rng(seed)

    def _log_reward(self, mask: int, success: bool) -> float:
        if not success:
            return math.log(1e-4)            # tiny floor so non-witness terminals are near-zero mass
        if self.reward == "success":
            return 0.0                        # log 1
        return bin(mask).count("1") * math.log(self.p)   # log nu(up S) = |S| log p

    def _sample(self, context, *, greedy: bool = False):
        obs, am = self.env.reset(context)
        obss, ams, acts = [], [], []
        done, mask, success, canon = False, 0, False, 0
        while not done:
            ot = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mt = torch.as_tensor(np.asarray(am, dtype=bool), device=self.device).unsqueeze(0)
            dist = self.pf.distribution(ot, mt)
            a = int(dist.probs.argmax(-1).item()) if greedy else int(dist.sample().item())
            obss.append(np.asarray(obs, np.float32))
            ams.append(np.asarray(am, bool))
            acts.append(a)
            res = self.env.step(a)
            obs, am, done = res.obs, res.action_mask, res.done
            if done:
                mask, success = res.witness, res.success
                canon = int(res.info.get("canon_mask", res.witness))
        return obss, ams, acts, mask, success, canon

    def train(self, *, updates: int = 150, batch: int = 16) -> "GFlowNet":
        contexts = list(self.env.contexts)
        for _ in range(updates):
            residuals = []
            for _ in range(batch):
                ctx = contexts[int(self.rng.integers(len(contexts)))]
                obss, ams, acts, mask, success, _ = self._sample(ctx)
                ot = torch.as_tensor(np.asarray(obss), dtype=torch.float32, device=self.device)
                mt = torch.as_tensor(np.asarray(ams), dtype=torch.bool, device=self.device)
                logpf = self.pf.distribution(ot, mt).log_prob(
                    torch.as_tensor(acts, dtype=torch.long, device=self.device)).sum()
                k = bin(mask).count("1")
                log_pb = -sum(math.log(t) for t in range(1, k + 1))     # uniform P_B = -log(k!)
                residuals.append(self.logZ + logpf - self._log_reward(mask, success) - log_pb)
            loss = torch.stack(residuals).pow(2).mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.pf.parameters()) + [self.logZ], 1.0)
            self.opt.step()
        return self

    @torch.no_grad()
    def rollout(self, context) -> Rollout:
        obss, ams, acts, mask, success, canon = self._sample(context)
        return Rollout(obss, ams, acts, [], mask, canon, success, float(success))


class PenalizedGFlowNet(GFlowNet):
    """Trajectory-balance GFlowNet with log R(S) = -lam |S| on witnesses.

    lam = 0 is the success variant and lam = log(1/p) the up-set variant. A failed terminal keeps the
    log 1e-4 floor wherever that lies below the smallest witness reward, and otherwise sits 100x below
    it. ``logz_lr`` gives log Z its own Adam learning rate, the usual trajectory-balance setup; ``None``
    keeps one learning rate (3e-3) for the policy and log Z.
    """

    def __init__(self, env: MaxSatEnv, *, lam: float, logz_lr: float | None = None, seed: int = 0) -> None:
        super().__init__(env, reward="upset", p=0.7, seed=seed)
        self.lam = lam
        self.floor = min(math.log(1e-4), -lam * env.horizon - math.log(100.0))
        if logz_lr is not None:
            self.opt = torch.optim.Adam([{"params": list(self.pf.parameters()), "lr": 3e-3},
                                         {"params": [self.logZ], "lr": logz_lr}])

    def _log_reward(self, mask: int, success: bool) -> float:
        if not success:
            return self.floor
        return (-self.lam) * bin(mask).count("1")
