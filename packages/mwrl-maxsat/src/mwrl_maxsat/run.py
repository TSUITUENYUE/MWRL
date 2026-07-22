"""Prime-implicant enumeration benchmark on the amortized MWRL library.

Every method runs through the core ``Runner`` over a ``MaxSatEnv`` (the add-a-variable
MDP); the benchmark supplies only the env and the exact-antichain scoring. MWRL uses the
up-set credit with REINFORCE, and PPO / MaxRL / MaxRL+size run as other (advantage,
optimizer) settings of the same runner. Per instance the context set is a singleton, so
the amortized runner reproduces the paper's per-instance rows; pass a multi-instance
context set for cross-instance amortization.

    python -m mwrl_maxsat.run --instances 8 --updates 150
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from mwrl.config import MWRLConfig
from mwrl.runner import Runner

from mwrl_maxsat.env import MaxSatEnv
from mwrl_maxsat.evaluate import evaluate
from mwrl_maxsat.instance import generate_with_antichain

# (name, advantage, optimizer, witness_baseline|None, size_penalty)
CONFIGS = [
    ("PPO", "maxrl", "ppo", None, 0.0),
    ("MaxRL", "maxrl", "reinforce", None, 0.0),
    ("MaxRL+size", "maxrl_size", "reinforce", None, 0.1),
    ("MWRL (mean)", "mwrl", "reinforce", "mean", 0.0),
    ("MWRL (l2o)", "mwrl", "reinforce", "l2o", 0.0),
]


def summarize(name, metrics):
    def m(k):
        return np.mean([x[k] for x in metrics])
    print(f"{name:<16} soundness {m('soundness'):.2f}  completeness {m('completeness'):.2f}  "
          f"#found {m('distinct_born_minimal'):4.1f}/{m('antichain_size'):.0f}  "
          f"raw|S| {m('raw_median_size'):4.1f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--n-vars", type=int, default=14)
    ap.add_argument("--n-clauses", type=int, default=10)
    ap.add_argument("--clause-len", type=int, default=3)
    ap.add_argument("--min-modes", type=int, default=6)
    ap.add_argument("--max-modes", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=12)
    ap.add_argument("--groups-per-update", type=int, default=8)  # task_batch_size for a singleton context
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--allow-remove", action="store_true", help="add a remove action ({add,remove,stop})")
    ap.add_argument("--canonicalize", action="store_true",
                    help="count valuation: prune each witness to a minimal Phi (randomized order)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--nu", default="layer", choices=["layer", "uniform", "geometric"],
                    help="coverage measure (geometric enables the MC path at large groups)")
    ap.add_argument("--nu-p", type=float, default=0.7, help="geometric measure prior p")
    ap.add_argument("--mc-samples", type=int, default=20000,
                    help="coverage union-measure Monte-Carlo samples above the exact threshold")
    ap.add_argument("--eval-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    instances = []
    for i in range(args.instances):
        rng = np.random.default_rng(1000 + i)
        instances.append(generate_with_antichain(
            args.n_vars, args.n_clauses, args.clause_len,
            min_modes=args.min_modes, max_modes=args.max_modes, rng=rng))
    sizes = [len(ac) for _, ac in instances]
    print(f"[data] {len(instances)} instances, n_vars={args.n_vars}, "
          f"|M| mean={np.mean(sizes):.1f} (range {min(sizes)}-{max(sizes)})\n", flush=True)

    def make_config(baseline):
        common = dict(updates=args.updates, task_batch_size=args.groups_per_update,
                      group_size=args.group_size, learning_rate=args.lr, entropy_coef=args.entropy,
                      hidden_dim=128, update_epochs=1, minibatch_size=256, clip_epsilon=0.2,
                      max_grad_norm=1.0, epsilon=1e-6)
        if baseline is None:
            return MWRLConfig(**common)
        valuation = "count" if args.canonicalize else "coverage"
        return MWRLConfig(witness_valuation=valuation, witness_center=True, witness_baseline=baseline,
                          witness_normalize=True, witness_scale="mean", witness_nu=args.nu,
                          witness_nu_p=args.nu_p, witness_mc_samples=args.mc_samples, **common)

    torch.manual_seed(args.seed)
    print(f"{'method':<16} soundness  completeness  #found=distinct minimal witnesses")
    print("-" * 78)
    for name, advantage, optimizer, baseline, size_penalty in CONFIGS:
        config = make_config(baseline)
        metrics = []
        for inst, ac in instances:
            env = MaxSatEnv([inst], d_max=args.n_vars, horizon=args.horizon,
                            allow_remove=args.allow_remove, canonicalize_witness=args.canonicalize,
                            seed=args.seed)
            runner = Runner(env, config, advantage=advantage, optimizer=optimizer,
                            size_penalty=size_penalty, seed=args.seed)
            runner.train()
            metrics.append(evaluate(runner, inst, ac, n_samples=args.eval_samples))
        summarize(name, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
