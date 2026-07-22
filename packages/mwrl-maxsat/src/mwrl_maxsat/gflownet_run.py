"""Reproducible driver for the two GFlowNet baseline rows of the MaxSAT table.

Trains the trajectory-balance GFlowNet (``gflownet.py``) under both reward variants
(success and up-set) on the SAME eight instances as ``table.py`` (instance rng 1000+i),
for several policy seeds, and scores each run with the shared exact-antichain scorer
(``evaluate``). Reports per-variant mean and standard deviation across seeds after
averaging the instances, matching the aggregation of the learned rows in the main table.

    python -m mwrl_maxsat.gflownet_run          # seeds 0,1,2 -> runs/maxsat/gflownet_seeds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mwrl_maxsat.env import MaxSatEnv
from mwrl_maxsat.evaluate import evaluate
from mwrl_maxsat.gflownet import GFlowNet
from mwrl_maxsat.instance import generate_with_antichain

VARIANTS = ("success", "upset")
DEF = dict(instances=8, n_vars=14, n_clauses=10, clause_len=3, min_modes=6, max_modes=20,
           updates=150, batch=16, eval=128, p=0.7, seeds="0,1,2")


def _worker(args):
    variant, seed, ii, cfg = args
    torch.set_num_threads(1)
    inst, ac = generate_with_antichain(cfg["n_vars"], cfg["n_clauses"], cfg["clause_len"],
                                       min_modes=cfg["min_modes"], max_modes=cfg["max_modes"],
                                       rng=np.random.default_rng(1000 + ii))
    env = MaxSatEnv([inst], d_max=cfg["n_vars"], horizon=12,
                    canonicalize_witness=False, seed=seed)   # raw witnesses: the coverage view
    gfn = GFlowNet(env, reward=variant, p=cfg["p"], seed=seed)
    gfn.train(updates=cfg["updates"], batch=cfg["batch"])
    return variant, seed, evaluate(gfn, inst, ac, n_samples=cfg["eval"])


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in DEF.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--output", default="runs/maxsat/gflownet_seeds.json")
    cfg = vars(ap.parse_args())
    out = Path(cfg.pop("output"))
    seeds = [int(s) for s in cfg["seeds"].split(",")]

    jobs = [(v, s, i, cfg) for v in VARIANTS for s in seeds for i in range(cfg["instances"])]
    import os
    from multiprocessing import Pool
    nproc = min(os.cpu_count() or 4, len(jobs))
    print(f"[parallel] {len(jobs)} runs / {nproc} procs  updates={cfg['updates']} "
          f"batch={cfg['batch']} seeds={seeds}", flush=True)
    with Pool(nproc) as pool:
        res = pool.map(_worker, jobs)

    report: dict = {"config": cfg, "per_seed": {}, "aggregate": {}}
    for variant in VARIANTS:
        seed_means = {}
        for s in seeds:
            ms = [m for v, sd, m in res if v == variant and sd == s]
            seed_means[s] = {k: float(np.mean([m[k] for m in ms]))
                             for k in ("soundness", "completeness", "distinct_born_minimal",
                                       "raw_median_size", "success_rate")}
        report["per_seed"][variant] = seed_means
        agg = {k: (float(np.mean([seed_means[s][k] for s in seeds])),
                   float(np.std([seed_means[s][k] for s in seeds])))
               for k in next(iter(seed_means.values()))}
        report["aggregate"][variant] = agg
        print(f"\n=== GFlowNet ({variant}) mean±std over seeds {seeds} ===")
        for k, (m, sd) in agg.items():
            print(f"  {k:22s} {m:6.3f} ± {sd:.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[saved] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
