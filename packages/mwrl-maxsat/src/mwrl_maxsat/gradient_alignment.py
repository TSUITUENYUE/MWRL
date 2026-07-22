"""Gradient-level validation of the shared-sample Monte Carlo credit.

The credit-vector cosine (credit_validation.py) does not directly bound what training
sees: the policy update is g = sum_i c_i * grad log pi(tau_i). This script freezes a
briefly trained policy, samples groups from it, computes the exact full-lattice credit
and the shared-sample MC credit on the SAME groups, and reports the cosine between the
two assembled policy gradients, per sample size N.

    python -m mwrl_maxsat.gradient_alignment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mwrl.config import MWRLConfig
from mwrl.credit import coverage_l2o_advantages, nu_upset_geometric
from mwrl.runner import Runner
from mwrl_maxsat.credit_validation import exact_credit
from mwrl_maxsat.env import MaxSatEnv
from mwrl_maxsat.instance import generate_with_antichain


def _policy_gradient(runner: Runner, rolls, credits) -> np.ndarray:
    """Flattened sum_i c_i * grad log pi(tau_i) at the current policy parameters."""
    runner.actor.zero_grad(set_to_none=True)
    total = None
    for roll, credit in zip(rolls, credits, strict=True):
        if credit == 0.0:
            continue
        logp = torch.zeros((), dtype=torch.float32)
        for obs, mask, action in zip(roll.obs, roll.action_masks, roll.actions, strict=True):
            ot = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mt = torch.as_tensor(np.asarray(mask, dtype=bool)).unsqueeze(0)
            dist = runner.actor.distribution(ot, mt)
            logp = logp + dist.log_prob(torch.as_tensor([action]))[0]
        grads = torch.autograd.grad(float(credit) * logp, list(runner.actor.parameters()))
        flat = torch.cat([g.reshape(-1) for g in grads]).detach().numpy()
        total = flat if total is None else total + flat
    return total if total is not None else np.zeros(1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vars", type=int, default=14)
    ap.add_argument("--group", type=int, default=48)
    ap.add_argument("--train-updates", type=int, default=40)
    ap.add_argument("--groups", type=int, default=12)
    ap.add_argument("--p", type=float, default=0.7)
    ap.add_argument("--samples", default="256,1024,5000,20000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="runs/maxsat_validations")
    args = ap.parse_args()

    inst, ac = generate_with_antichain(args.n_vars, 10, 3, min_modes=6, max_modes=20,
                                       rng=np.random.default_rng(1000))
    env = MaxSatEnv([inst], d_max=args.n_vars, horizon=12, seed=args.seed)
    torch.manual_seed(args.seed)
    cfg = MWRLConfig(updates=args.train_updates, task_batch_size=8, group_size=args.group,
                     learning_rate=3e-3, entropy_coef=0.02, hidden_dim=128, update_epochs=1,
                     minibatch_size=256, clip_epsilon=0.2, max_grad_norm=1.0, epsilon=1e-6,
                     witness_valuation="coverage", witness_center=True, witness_baseline="l2o",
                     witness_normalize=True, witness_scale="mean", witness_nu="geometric",
                     witness_nu_p=args.p, witness_mc_samples=20000)
    runner = Runner(env, cfg, advantage="mwrl", optimizer="reinforce", seed=args.seed)
    runner.train()

    nu = nu_upset_geometric(args.p)
    sample_grid = [int(x) for x in args.samples.split(",")]
    rows = {n: [] for n in sample_grid}
    for g in range(args.groups):
        rolls = [runner.rollout(inst) for _ in range(args.group)]
        masks = [r.witness for r in rolls]
        succ = [r.success for r in rolls]
        _, exact = exact_credit(masks, succ, n_vars=args.n_vars, p=args.p)
        g_exact = _policy_gradient(runner, rolls, list(exact))
        for n in sample_grid:
            mc = coverage_l2o_advantages(masks, succ, nu_upset=nu, mc_p=args.p,
                                          mc_samples=n, mc_threshold=0,
                                          rng=np.random.default_rng(10_000 + 97 * g + n))
            g_mc = _policy_gradient(runner, rolls, mc)
            rows[n].append(_cosine(g_exact, g_mc))

    print(f"{'N':>8}  {'gradient cosine (mean +/- sd over groups)':<44}")
    summary = {}
    for n in sample_grid:
        vals = np.asarray(rows[n])
        summary[n] = {"mean": float(vals.mean()), "sd": float(vals.std())}
        print(f"{n:>8}  {vals.mean():.3f} +/- {vals.std():.3f}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gradient_alignment.json").write_text(json.dumps(
        {"config": vars(args), "antichain": len(ac), "cosines": summary}, indent=2))
    print(f"[saved] {out / 'gradient_alignment.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
