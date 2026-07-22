"""The amortized MWRL runner: grouped score-function policy gradient with the up-set
leave-one-out credit.

The credit is ``A_i = V(U_K) - V(U_{-i})`` (theory 4.4), plugged into the score-function
estimator ``grad J = E[sum_i A_i grad log P(tau_i | c)]`` (theory 4.2). That estimator is
REINFORCE, so ``reinforce`` is the default optimizer; ``ppo`` adds clipping for the PPO
baseline and reduces to plain REINFORCE at ``clip_epsilon = 0``. The advantage axis
(``mwrl`` or the scalar ``maxrl``/``grpo``/``rloo``) and the optimizer axis (``reinforce``
or ``ppo``) vary independently. Amortization is the outer loop: each update samples
``c ~ D`` and trains one policy ``pi_theta(.|c)`` over the whole distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mwrl.advantage import mwrl_group_advantages
from mwrl.baselines import grouped_outcome_advantages
from mwrl.config import GroupedRLConfig
from mwrl.credit import popcount
from mwrl.env import MWRLEnv
from mwrl.modules import Actor


def normalize_update_advantages(
    advantages: list[float], *, scale: str, epsilon: float
) -> list[float]:
    """Apply one positive scale to every MWRL advantage in an optimizer update.

    Unlike per-group normalization, this preserves the direction of the realized
    aggregate policy-gradient term because every group receives the same divisor.
    """
    values = torch.as_tensor(advantages, dtype=torch.float32)
    divisor = values.abs().mean() if scale == "mean" else values.std(unbiased=False)
    return (values / (divisor + epsilon)).tolist()


@dataclass
class Rollout:
    obs: list[np.ndarray]
    action_masks: list[np.ndarray]
    actions: list[int]
    log_probs: list[float]      # sampling-time log-probs (used only by the PPO baseline)
    witness: int
    canon: int
    success: bool
    reward: float


class Runner:
    def __init__(self, env: MWRLEnv, config: GroupedRLConfig, *, advantage: str = "mwrl",
                 optimizer: str = "reinforce", size_penalty: float = 0.0,
                 device: str = "cpu", seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.env = env
        self.config = config
        self.advantage = advantage        # "mwrl" | "maxrl" | "grpo" | "rloo" | "maxrl_size"
        self.optimizer = optimizer        # "reinforce" (default, = the method) | "ppo" (baseline)
        self.size_penalty = size_penalty
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.actor = Actor(env.num_obs, env.num_actions, hidden_dim=config.hidden_dim).to(self.device)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.history: list[dict] = []

    def rollout(self, context) -> Rollout:
        obs, mask = self.env.reset(context)
        obss: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        logps: list[float] = []
        done, witness, canon, success, reward = False, 0, 0, False, 0.0
        while not done:
            ot = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mt = torch.as_tensor(np.asarray(mask, dtype=bool), device=self.device).unsqueeze(0)
            with torch.no_grad():
                dist = self.actor.distribution(ot, mt)
                a = dist.sample()
            obss.append(np.asarray(obs, dtype=np.float32))
            masks.append(np.asarray(mask, dtype=bool))
            actions.append(int(a.item()))
            logps.append(float(dist.log_prob(a).item()))
            res = self.env.step(int(a.item()))
            obs, mask, reward, done = res.obs, res.action_mask, res.reward, res.done
            if done:
                witness, success = res.witness, res.success
                canon = int(res.info.get("canon_mask", res.witness))
        return Rollout(obss, masks, actions, logps, witness, canon, success, float(reward))

    def _rollout_group(self, context, k: int) -> list[Rollout]:
        """One context's group of ``k`` rollouts. When the env supports a deferred terminal and the
        context a batched verifier (the circuits path), the ``k`` terminal verifications run as ONE
        batched inner-optimizer search instead of ``k`` separate ones; else sequential fallback."""
        if not (hasattr(self.env, "defer_terminal") and hasattr(context, "s_exists_batch")):
            return [self.rollout(context) for _ in range(k)]
        self.env.defer_terminal = True
        try:
            rolls = [self.rollout(context) for _ in range(k)]          # trajectories, terminals unverified
        finally:
            self.env.defer_terminal = False
        for r, (found, canon) in zip(rolls, context.s_exists_batch([r.witness for r in rolls]),
                                     strict=True):
            r.success, r.canon, r.reward = found, (canon if canon else r.witness), float(found)
        return rolls

    def _group_advantages(self, group: list[Rollout]) -> list[float]:
        successes = [r.success for r in group]
        if self.advantage == "mwrl":
            adv = mwrl_group_advantages(
                terminal_masks=[r.witness for r in group], canon_masks=[r.canon for r in group],
                successes=successes, rewards=[r.reward for r in group],
                config=self.config, device=self.device)
        elif self.advantage == "maxrl_size":  # scalar minimality: reward = success - lambda*|S|
            rewards = [float(s) - self.size_penalty * popcount(r.witness)
                       for s, r in zip(successes, group, strict=True)]
            rew = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            adv = (rew - rew.mean()) / (rew.std() + self.config.epsilon)
        else:
            rew = torch.as_tensor([r.reward for r in group], dtype=torch.float32, device=self.device)
            adv = grouped_outcome_advantages(rew, mode=self.advantage, epsilon=self.config.epsilon)
        return adv.detach().cpu().tolist()

    def train(self, on_update=None) -> Actor:
        contexts = list(self.env.contexts)
        for update in range(self.config.updates):
            items: list[tuple[Rollout, float]] = []
            # amortize over c ~ D: task_batch_size distinct contexts per update, or (when
            # it exceeds the pool, e.g. a single-instance run) that many groups with repeats.
            n_ctx = len(contexts)
            idx = self.rng.choice(n_ctx, size=self.config.task_batch_size,
                                  replace=self.config.task_batch_size > n_ctx)
            for i in idx:  # amortize: each sampled context gets its own group + credit
                group = self._rollout_group(contexts[int(i)], self.config.group_size)
                items.extend(zip(group, self._group_advantages(group), strict=True))
            if items:
                if (
                    self.advantage == "mwrl"
                    and self.config.witness_normalize
                    and self.config.witness_normalization_scope == "update"
                ):
                    normalized = normalize_update_advantages(
                        [advantage for _, advantage in items],
                        scale=self.config.witness_scale,
                        epsilon=self.config.epsilon,
                    )
                    items = [
                        (rollout, advantage)
                        for (rollout, _), advantage in zip(items, normalized, strict=True)
                    ]
                (self._ppo_update if self.optimizer == "ppo" else self._reinforce_update)(items)
                self.history.append({"update": update,
                                     "success_rate": sum(r.success for r, _ in items) / len(items)})
                if on_update is not None:
                    on_update(update, self.history[-1]["success_rate"])
        return self.actor

    def _reinforce_update(self, items: list[tuple[Rollout, float]]) -> None:
        """Score-function estimator: loss = -mean_i A_i * sum_t log pi(a_t|s_t) (theory 4.2)."""
        obs, masks, acts, traj_ids, advs = [], [], [], [], []
        for tid, (r, a) in enumerate(items):
            obs.extend(r.obs)
            masks.extend(r.action_masks)
            acts.extend(r.actions)
            traj_ids.extend([tid] * len(r.actions))
            advs.append(a)
        dist = self.actor.distribution(
            torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=self.device))
        logp = dist.log_prob(torch.as_tensor(acts, dtype=torch.long, device=self.device))
        logp_traj = torch.zeros(len(items), device=self.device).index_add_(
            0, torch.as_tensor(traj_ids, dtype=torch.long, device=self.device), logp)
        adv_t = torch.as_tensor(advs, dtype=torch.float32, device=self.device)
        loss = -(adv_t * logp_traj).mean() - self.config.entropy_coef * dist.entropy().mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.opt.step()

    def _ppo_update(self, items: list[tuple[Rollout, float]]) -> None:
        """Clipped PPO baseline (per-step). Set clip_epsilon -> 0 to recover un-ratioed steps."""
        obs, masks, acts, oldlp, advs = [], [], [], [], []
        for r, a in items:
            obs.extend(r.obs)
            masks.extend(r.action_masks)
            acts.extend(r.actions)
            oldlp.extend(r.log_probs)
            advs.extend([a] * len(r.actions))
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=self.device)
        act_t = torch.as_tensor(acts, dtype=torch.long, device=self.device)
        oldlp_t = torch.as_tensor(oldlp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advs, dtype=torch.float32, device=self.device)
        n = act_t.shape[0]
        for _ in range(self.config.update_epochs):
            for s in torch.randperm(n, device=self.device).split(self.config.minibatch_size):
                dist = self.actor.distribution(obs_t[s], mask_t[s])
                ratio = torch.exp(torch.clamp(dist.log_prob(act_t[s]) - oldlp_t[s], -20.0, 20.0))
                a = adv_t[s]
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                                        1 + self.config.clip_epsilon) * a)
                loss = -surr.mean() - self.config.entropy_coef * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.opt.step()
