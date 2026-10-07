"""Size-penalized MaxEnt RL and GFlowNet on the MaxSAT table's protocol.

Both families spread probability over sufficient sets, and a size penalty pushes them toward
small ones. Each family is swept over its penalty strength at the main table's budget of 57,600
terminal verifier calls per formula, so its reported row is its best setting.

maxent  MaxEnt RL: the core runner's ``maxent`` advantage, the group-standardized soft return
        s(S) - lam |S| - alpha sum_t log pi(a_t | s_t), optimized with the PPO row's settings.
        At alpha = 0 and lam = 0.1 it coincides with the PPO + size row.
gfn     Trajectory-balance GFlowNet with R(S) = s(S) exp(-lam |S|) (``PenalizedGFlowNet``), log Z on
        its own learning rate of 0.1, batch 16 for 3,600 updates.

    python -m mwrl_maxsat.size_penalty_sweep      # 30 settings, 8 formulas, seeds 0, 1, 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from mwrl.config import MWRLConfig
from mwrl.runner import Runner

from mwrl_maxsat.env import MaxSatEnv
from mwrl_maxsat.evaluate import evaluate
from mwrl_maxsat.gflownet import PenalizedGFlowNet
from mwrl_maxsat.instance import generate_with_antichain

# (alpha, lam) for MaxEnt RL and lam for the GFlowNet; every family's optimum lies inside its grid.
MAXENT_GRID = ([(0.0, 0.1)] + [(a, lam) for a in (0.02, 0.05, 0.1, 0.2) for lam in (0.0, 0.1, 0.2)]
               + [(a, lam) for a in (0.05, 0.1, 0.2) for lam in (0.3, 0.5, 1.0)])
GFN_GRID = [0.0, -math.log(0.7), -math.log(0.5), -math.log(0.3), -math.log(0.1), 0.9, 1.5, 1.8]
GFN_LOGZ_LR = 0.1
SEEDS = (0, 1, 2)
DEF = dict(instances=8, n_vars=14, n_clauses=10, clause_len=3, min_modes=6, max_modes=20,
           group=48, gpu=8, updates=150, eval=128, gfn_batch=16, gfn_updates=3600)
METRICS = ("soundness", "completeness", "distinct_born_minimal", "raw_median_size", "success_rate")


def _ppo_config(cfg: dict) -> MWRLConfig:
    """The PPO + size row's configuration."""
    return MWRLConfig(updates=cfg["updates"], task_batch_size=cfg["gpu"], group_size=cfg["group"],
                      learning_rate=3e-3, entropy_coef=0.02, hidden_dim=128, update_epochs=1,
                      minibatch_size=256, clip_epsilon=0.2, max_grad_norm=1.0, epsilon=1e-6)


def _worker(job: dict) -> dict:
    torch.set_num_threads(1)
    cfg, seed, ii = job["cfg"], job["seed"], job["instance"]
    inst, ac = generate_with_antichain(cfg["n_vars"], cfg["n_clauses"], cfg["clause_len"],
                                       min_modes=cfg["min_modes"], max_modes=cfg["max_modes"],
                                       rng=np.random.default_rng(1000 + ii))
    env = MaxSatEnv([inst], d_max=cfg["n_vars"], horizon=12, canonicalize_witness=False, seed=seed)
    torch.manual_seed(seed)
    if job["family"] == "maxent":
        model = Runner(env, _ppo_config(cfg), advantage="maxent", optimizer="ppo",
                       size_penalty=job["lam"], maxent_alpha=job["alpha"], seed=seed)
        model.train()
    else:
        model = PenalizedGFlowNet(env, lam=job["lam"], logz_lr=GFN_LOGZ_LR, seed=seed)
        model.train(updates=cfg["gfn_updates"], batch=cfg["gfn_batch"])
    m = evaluate(model, inst, ac, n_samples=cfg["eval"])
    return {k: job[k] for k in ("family", "alpha", "lam", "seed", "instance")} | {k: m[k] for k in METRICS}


def _aggregate(res: list[dict]) -> list[dict]:
    """Mean and standard deviation over seeds of each seed's mean over the formulas."""
    rows = []
    settings = sorted({(r["family"], r["alpha"], r["lam"]) for r in res},
                      key=lambda c: (c[0], -1 if c[1] is None else c[1], c[2]))
    for fam, a, lam in settings:
        per_seed = [{k: float(np.mean([r[k] for r in res
                                       if (r["family"], r["alpha"], r["lam"], r["seed"]) == (fam, a, lam, s)]))
                     for k in METRICS} for s in SEEDS]
        rows.append(dict(family=fam, alpha=a, lam=lam,
                         **{k: [float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed]))]
                            for k in METRICS}))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in DEF.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--output", default="runs/maxsat")
    cfg = vars(ap.parse_args())
    out = Path(cfg.pop("output"))
    print("[resolved-config] maxent " + json.dumps(_ppo_config(cfg).model_dump()), flush=True)

    jobs = [dict(family="maxent", alpha=a, lam=lam, seed=s, instance=i, cfg=cfg)
            for a, lam in MAXENT_GRID for s in SEEDS for i in range(cfg["instances"])]
    jobs += [dict(family="gfn", alpha=None, lam=lam, seed=s, instance=i, cfg=cfg)
             for lam in GFN_GRID for s in SEEDS for i in range(cfg["instances"])]
    nproc = min(os.cpu_count() or 4, len(jobs))
    print(f"[parallel] {len(jobs)} runs / {nproc} procs", flush=True)
    with Pool(nproc) as pool:
        res = pool.map(_worker, jobs)

    rows = _aggregate(res)
    print(f"\n{'setting':<28}{'#found':>12}{'born-min':>10}{'recall':>8}{'size':>6}")
    print("-" * 64)
    for r in rows:
        tag = (f"maxent a={r['alpha']} lam={r['lam']}" if r["family"] == "maxent"
               else f"gfn lam={r['lam']:.3f}")
        print(f"{tag:<28}{r['distinct_born_minimal'][0]:7.2f}±{r['distinct_born_minimal'][1]:.2f}"
              f"{r['soundness'][0]:10.3f}{r['completeness'][0]:8.3f}{r['raw_median_size'][0]:6.1f}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "size_penalty_sweep.json").write_text(json.dumps(
        {"config": cfg, "seeds": list(SEEDS), "maxent_grid": MAXENT_GRID, "gfn_grid": GFN_GRID,
         "aggregate": rows, "runs": res}, indent=2))
    print(f"\n[saved] {out / 'size_penalty_sweep.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
