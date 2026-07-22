"""Normalization ablation: witness_scale mean vs std, everything else identical.

Every experiment package has overridden the core default (std) with witness_scale="mean"
since the benchmark migration. This runs the MWRL rows of the paper table under BOTH
scales on the SAME instances, seeds, and budget, so the comparison is paired: if the
difference is small, mean-scale stands as the documented recipe; if it is large, the
affected tables get rerun.

    python -m mwrl_maxsat.scale_ablation       # group=48, p=0.7, 150 updates, 8 instances
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from mwrl.config import MWRLConfig
from mwrl.runner import Runner

from mwrl_maxsat.env import MaxSatEnv
from mwrl_maxsat.evaluate import evaluate
from mwrl_maxsat.instance import generate_with_antichain

CONFIGS = [("MWRL (mean ctr)", "mean"), ("MWRL (l2o)", "l2o")]   # (name, witness_baseline)
SCALES = ("mean", "std")
DEF = dict(instances=8, n_vars=14, n_clauses=10, clause_len=3, min_modes=6, max_modes=20,
           group=48, gpu=8, updates=150, eval=128, mc=20000, p=0.7, seed=0)


def _make_config(baseline: str, valuation: str, scale: str, cfg: dict) -> MWRLConfig:
    return MWRLConfig(witness_valuation=valuation, witness_center=True,
                      witness_baseline=baseline, witness_normalize=True, witness_scale=scale,
                      witness_nu="geometric", witness_nu_p=cfg["p"],
                      witness_mc_samples=cfg["mc"], updates=cfg["updates"],
                      task_batch_size=cfg["gpu"], group_size=cfg["group"], learning_rate=3e-3,
                      entropy_coef=0.02, hidden_dim=128, update_epochs=1, minibatch_size=256,
                      clip_epsilon=0.2, max_grad_norm=1.0, epsilon=1e-6)


def _worker(args):
    valuation, scale, ci, ii, cfg = args
    torch.set_num_threads(1)
    baseline = CONFIGS[ci][1]
    inst, ac = generate_with_antichain(cfg["n_vars"], cfg["n_clauses"], cfg["clause_len"],
                                       min_modes=cfg["min_modes"], max_modes=cfg["max_modes"],
                                       rng=np.random.default_rng(1000 + ii))
    env = MaxSatEnv([inst], d_max=cfg["n_vars"], horizon=12,
                    canonicalize_witness=(valuation == "count"), seed=cfg["seed"])
    torch.manual_seed(cfg["seed"])
    r = Runner(env, _make_config(baseline, valuation, scale, cfg), advantage="mwrl",
               optimizer="reinforce", seed=cfg["seed"])
    r.train()
    m = evaluate(r, inst, ac, n_samples=cfg["eval"])
    return valuation, scale, ci, ii, m


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in DEF.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--output", default="runs/maxsat")
    cfg = vars(ap.parse_args())
    out = Path(cfg.pop("output"))
    print("[resolved-config] " + json.dumps(
        _make_config("l2o", "coverage", "std", cfg).model_dump()), flush=True)

    jobs = [(val, sc, ci, ii, cfg) for val in ("coverage", "count") for sc in SCALES
            for ci in range(len(CONFIGS)) for ii in range(cfg["instances"])]
    nproc = min(os.cpu_count() or 4, len(jobs))
    print(f"[parallel] {len(jobs)} runs / {nproc} procs  group={cfg['group']} "
          f"updates={cfg['updates']}", flush=True)
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        res = pool.map(_worker, jobs)

    table: dict = {}
    print(f"\n{'valuation':<10}{'baseline':<12}{'scale':<7}{'sound':>7}{'compl':>7}{'#found':>8}")
    print("-" * 51)
    for val in ("coverage", "count"):
        for ci, (name, baseline) in enumerate(CONFIGS):
            for sc in SCALES:
                ms = {ii: m for v, s, c, ii, m in res if (v, s, c) == (val, sc, ci)}
                row = dict(soundness=float(np.mean([m["soundness"] for m in ms.values()])),
                           completeness=float(np.mean([m["completeness"] for m in ms.values()])),
                           found=float(np.mean([m["distinct_born_minimal"] for m in ms.values()])),
                           per_instance={str(k): {"soundness": m["soundness"],
                                                  "completeness": m["completeness"]}
                                         for k, m in ms.items()})
                table[f"{val}/{baseline}/{sc}"] = row
                print(f"{val:<10}{baseline:<12}{sc:<7}{row['soundness']:7.2f}"
                      f"{row['completeness']:7.2f}{row['found']:8.1f}", flush=True)
            a, b = (table[f"{val}/{baseline}/{s}"] for s in SCALES)
            print(f"{'':<10}{'':<12}{'diff':<7}{a['soundness'] - b['soundness']:+7.2f}"
                  f"{a['completeness'] - b['completeness']:+7.2f}"
                  f"{a['found'] - b['found']:+8.1f}   (mean - std, paired)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale_ablation.json").write_text(json.dumps({"config": cfg, "table": table}, indent=2))
    print(f"\n[saved] {out / 'scale_ablation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
