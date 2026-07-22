"""The full MaxSAT diagnostic table: every method under BOTH valuations (count and coverage), with
the two value-iteration ceilings, saved to ``runs/maxsat/``.

Two corners of the trilemma sit side by side:
  * coverage -- certificate-free (no canonicalizer); the up-set measure nu is the reward, so the
    POLICY must reach minimal witnesses itself. Discriminative: scalar methods find nothing.
  * count    -- needs the canonicalizer Phi (the env prunes each terminal to a minimal witness);
    the credit counts distinct Phi. Less discriminative: the canonicalizer does the minimality,
    so even a scalar policy's terminals become minimal (soundness 1.0 for all).

Two ceilings bound the table:
  * set-language value iteration -- the fixed point is the antichain up-set ↑M (completeness 1.0).
  * vanilla scalar value iteration -- the fixed point is one optimal witness (completeness 1/|M|),
    the ceiling every scalar objective (PPO/MaxRL/GRPO/RLOO/MaxRL+size) is capped by.

    python -m mwrl_maxsat.table            # group=48, geom p=0.7, 150 updates, 8 instances
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
from mwrl.credit import nu_upset_geometric, union_upset_measure
from mwrl.runner import Runner

from mwrl_maxsat.env import MaxSatEnv
from mwrl_maxsat.evaluate import evaluate
from mwrl_maxsat.instance import generate_with_antichain

CONFIGS = [                                   # (name, advantage, optimizer, witness_baseline, size)
    ("PPO", "maxrl", "ppo", None, 0.0),
    ("MaxRL", "maxrl", "reinforce", None, 0.0),
    ("GRPO", "grpo", "reinforce", None, 0.0),
    ("RLOO", "rloo", "reinforce", None, 0.0),
    ("MaxRL+size", "maxrl_size", "reinforce", None, 0.1),
    ("MWRL (mean)", "mwrl", "reinforce", "mean", 0.0),
    ("MWRL (l2o)", "mwrl", "reinforce", "l2o", 0.0),
]
DEF = dict(instances=8, n_vars=14, n_clauses=10, clause_len=3, min_modes=6, max_modes=20,
           group=48, gpu=8, updates=150, eval=128, mc=20000, p=0.7, seed=0,
           vi_artifact="runs/maxsat_vi/executed_vi.json")


def _make_config(baseline, valuation, cfg):
    common = dict(updates=cfg["updates"], task_batch_size=cfg["gpu"], group_size=cfg["group"],
                  learning_rate=3e-3, entropy_coef=0.02, hidden_dim=128, update_epochs=1,
                  minibatch_size=256, clip_epsilon=0.2, max_grad_norm=1.0, epsilon=1e-6)
    if baseline is None:
        return MWRLConfig(**common)
    return MWRLConfig(witness_valuation=valuation, witness_center=True, witness_baseline=baseline,
                      witness_normalize=True, witness_scale="mean", witness_nu="geometric",
                      witness_nu_p=cfg["p"], witness_mc_samples=cfg["mc"], **common)


def _worker(args):
    valuation, ci, ii, cfg = args
    torch.set_num_threads(1)
    _, adv, opt, baseline, sp = CONFIGS[ci]
    inst, ac = generate_with_antichain(cfg["n_vars"], cfg["n_clauses"], cfg["clause_len"],
                                       min_modes=cfg["min_modes"], max_modes=cfg["max_modes"],
                                       rng=np.random.default_rng(1000 + ii))
    env = MaxSatEnv([inst], d_max=cfg["n_vars"], horizon=12,
                    canonicalize_witness=(valuation == "count"), seed=cfg["seed"])
    torch.manual_seed(cfg["seed"])
    r = Runner(env, _make_config(baseline, valuation, cfg), advantage=adv, optimizer=opt,
               size_penalty=sp, seed=cfg["seed"])
    r.train()
    return valuation, ci, evaluate(r, inst, ac, n_samples=cfg["eval"])


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in DEF.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--output", default="runs/maxsat")
    cfg = vars(ap.parse_args())
    out = Path(cfg.pop("output"))

    acs = [generate_with_antichain(cfg["n_vars"], cfg["n_clauses"], cfg["clause_len"],
                                   min_modes=cfg["min_modes"], max_modes=cfg["max_modes"],
                                   rng=np.random.default_rng(1000 + i))[1] for i in range(cfg["instances"])]
    meanM = float(np.mean([len(ac) for ac in acs]))
    nu = nu_upset_geometric(cfg["p"])
    v_full = float(np.mean([union_upset_measure(list(ac), nu, mc_p=cfg["p"], mc_samples=cfg["mc"],
                                                mc_threshold=12, rng=np.random.default_rng(0)) for ac in acs]))
    # the two planner rows come from the executed artifact, not analytical insertion
    artifact_path = Path(cfg.pop("vi_artifact"))
    if not artifact_path.exists():
        raise SystemExit(f"executed-VI artifact missing: {artifact_path}; "
                         "run `python -m mwrl_maxsat.vi` first")
    vi_runs = json.loads(artifact_path.read_text())["results"]
    if len(vi_runs) != cfg["instances"]:
        raise SystemExit("executed-VI artifact does not cover the table's instances")
    mw = [r["minimal_witness"] for r in vi_runs]
    sc = [r["scalar_penalized"] for r in vi_runs]
    ceilings = [
        ("Minimal-witness VI (executed)",
         float(np.mean([m["all_terminals_minimal"] for m in mw])),
         float(np.mean([m["final_recall"] for m in mw])),
         float(np.mean([m["found"] for m in mw])),
         float(np.mean([m["median_size"] for m in mw]))),
        ("Scalar VI (executed)",
         float(np.mean([s["witness_is_minimal"] for s in sc])),
         float(np.mean([s["completeness"] for s in sc])),
         float(np.mean([s["found_minimal"] for s in sc])),
         float(np.mean([s["witness_size"] for s in sc]))),
    ]

    jobs = [(val, ci, ii, cfg) for val in ("coverage", "count")
            for ci in range(len(CONFIGS)) for ii in range(cfg["instances"])]
    nproc = min(os.cpu_count() or 4, len(jobs))
    print(f"[parallel] {len(jobs)} runs / {nproc} procs  group={cfg['group']} updates={cfg['updates']} "
          f"|M|~{meanM:.0f}", flush=True)
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        res = pool.map(_worker, jobs)

    tables = {}
    for valuation in ("coverage", "count"):
        rows = [dict(method=n, soundness=s, completeness=c, found=f, antichain=meanM, raw_size=z)
                for n, s, c, f, z in ceilings]
        for ci, (name, *_r) in enumerate(CONFIGS):
            ms = [m for val, c, m in res if val == valuation and c == ci]
            rows.append(dict(method=name,
                             soundness=float(np.mean([m["soundness"] for m in ms])),
                             completeness=float(np.mean([m["completeness"] for m in ms])),
                             found=float(np.mean([m["distinct_born_minimal"] for m in ms])),
                             antichain=meanM,
                             raw_size=float(np.mean([m["raw_median_size"] for m in ms]))))
        tables[valuation] = rows
        print(f"\n=== {valuation.upper()} valuation "
              f"({'canonicalizer Phi' if valuation == 'count' else 'certificate-free'}) ===")
        print(f"{'method':<26}{'sound':>7}{'compl':>7}{'#found/|M|':>13}{'raw|S|':>8}")
        print("-" * 61)
        for j, r in enumerate(rows):
            print(f"{r['method']:<26}{r['soundness']:7.2f}{r['completeness']:7.2f}"
                  f"{r['found']:8.1f}/{r['antichain']:<4.0f}{r['raw_size']:8.1f}")
            if j == 1:
                print("-" * 61)
    scalar_size = ceilings[1][4]
    print(f"\ncoverage value V=nu(up M): set-language VI = {v_full:.3f}  "
          f"vanilla VI = {cfg['p'] ** scalar_size:.3f}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps({"config": cfg, "coverage_value_ceiling": v_full,
                                                "tables": tables}, indent=2))
    print(f"\n[saved] {out / 'table.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
