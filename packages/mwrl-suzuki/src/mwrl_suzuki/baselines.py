"""Amortization baselines for the Suzuki experiment (paper Table 2 / tab:amortization).

Rows match the table:
  * Per-substrate  -- a fresh policy retrained for each substrate; the recovery ceiling, no held-out.
  * MWRL           -- one fingerprint-conditioned policy amortized over substrates (the method).
  * Substrate-blind -- the same policy with the fingerprint zeroed (the amortization ablation).
  * Scalar RL      -- a scalar-reward (MaxRL) amortized policy.
All share the exact closed-world sufficiency verifier. Metrics: born-minimal rate, antichain recall on seen and
on held-out substrates, #found, and the total training wall-clock (its cost).

Every training here is independent and deterministic (fixed seeds), so the three amortized methods
and the 64 per-substrate policies all run as parallel jobs across the machine's cores -- identical
numbers to a sequential run, just using every core. Each worker loads the dataset once and pins BLAS
to a single thread so the pool does not oversubscribe.

    python -m mwrl_suzuki.baselines --holdout 13
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from mwrl_suzuki.schema import ConditionSchema

# ---------------------------------------------------------------------------------------------
# Per-worker dataset cache. The pool initializer loads the CSV + builds the verifiers ONCE per
# worker process; every job that process runs reuses them. Kept at module scope so it is inherited
# on fork / rebuilt on spawn.
_CACHE: dict = {}


def _init_worker(data_path: str, fp_method: str, n_bits: int, seed: int,
                 task_ids: tuple[str, ...] | None = None,
                 schema: ConditionSchema | None = None) -> None:
    # single-thread BLAS so N single-threaded workers do not fight over cores.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    from mwrl_suzuki.data import load_dataset

    # The parent already inferred the schema; re-deriving it here would re-materialize the whole
    # CSV in every worker (~2 GB each) to recover an object that pickles to under a kilobyte.
    if schema is None:
        schema = ConditionSchema.from_csv(data_path)
    # A per-substrate shard touches only its own tasks, so each worker holds that slice of the
    # grid instead of all 1,024 (the full load is ~6 GB, which caps a laptop at two workers).
    tasks = load_dataset(data_path, schema, fingerprint_method=fp_method, n_bits=n_bits,
                         task_ids=task_ids)
    _CACHE["schema"] = schema
    _CACHE["tasks"] = {t.task_id: t for t in tasks if t.antichain}
    _CACHE["seed"] = seed


def _verifier(task_id: str, *, blind: bool):
    from mwrl_suzuki.verifier import ChemVerifier

    v = ChemVerifier(_CACHE["tasks"][task_id], base_seed=_CACHE["seed"])
    if blind:
        v.attr_features = np.zeros_like(v.attr_features)  # substrate-blind: no fingerprint in the observation
    return v


def _agg(metrics: list[dict], key: str) -> float:
    return float(np.mean([m[key] for m in metrics])) if metrics else 0.0


def _amortized_job(payload: dict) -> dict:
    """One amortized policy over the training substrates; evaluated on seen + held-out."""
    from mwrl.runner import Runner

    from mwrl_suzuki.env import ChemEnv
    from mwrl_suzuki.evaluate import evaluate_antichain
    from mwrl_suzuki.run import build_config

    schema = _CACHE["schema"]
    n, seed, es = schema.n_dims, _CACHE["seed"], payload["eval_samples"]
    horizon = int(payload["mcfg"].get("horizon", n))
    vmap = {tid: _verifier(tid, blind=payload["blind"])
            for tid in payload["train_ids"] + payload["held_ids"]}
    fp_dim = len(next(iter(vmap.values())).attr_features)
    config = build_config(payload["mcfg"])

    t0 = time.time()
    env = ChemEnv([vmap[i] for i in payload["train_ids"]], n_comp=n, fp_dim=fp_dim, horizon=horizon)
    runner = Runner(env, config, advantage=payload["advantage"], optimizer="reinforce", seed=seed)
    runner.train()
    seen = [evaluate_antichain(runner, vmap[i], n_samples=es, record_proposals=True)
            for i in payload["train_ids"]]
    held = [evaluate_antichain(runner, vmap[i], n_samples=es, record_proposals=True)
            for i in payload["held_ids"]]
    per_task = [
        {
            "task_id": tid,
            "heldout": heldout,
            "completeness": m["completeness"],
            "soundness": m["soundness"],
            "n_found": m["n_found"],
            "antichain_size": len(vmap[tid].antichain),
            "antichain": m["antichain"],
            "proposals": m["proposals"],
        }
        for tid, heldout, m in (
            [(i, False, m) for i, m in zip(payload["train_ids"], seen)]
            + [(i, True, m) for i, m in zip(payload["held_ids"], held)]
        )
    ]
    return {
        "kind": "amortized", "name": payload["name"], "order": payload["order"],
        "born": _agg(held, "soundness"), "recall_seen": _agg(seen, "completeness"),
        "recall_held": _agg(held, "completeness"), "found": _agg(held, "n_found"),
        "tasks": per_task,
        "wall": time.time() - t0,
    }


def _persub_job(payload: dict) -> dict:
    """A fresh policy for a single substrate (the recovery-ceiling baseline)."""
    from mwrl.runner import Runner

    from mwrl_suzuki.env import ChemEnv
    from mwrl_suzuki.evaluate import evaluate_antichain
    from mwrl_suzuki.run import build_config

    schema = _CACHE["schema"]
    n, seed = schema.n_dims, _CACHE["seed"]
    horizon = int(payload["mcfg"].get("horizon", n))
    v = _verifier(payload["task_id"], blind=False)
    config = build_config({**payload["mcfg"], "updates": payload["per_updates"], "k": payload["per_k"]})

    t0 = time.time()
    env = ChemEnv([v], n_comp=n, fp_dim=len(v.attr_features), horizon=horizon)
    runner = Runner(env, config, advantage="mwrl", optimizer="reinforce", seed=seed)
    runner.train()
    m = evaluate_antichain(runner, v, n_samples=payload["eval_samples"], record_proposals=True)
    m["wall"] = time.time() - t0
    m["task_id"] = payload["task_id"]
    return {"kind": "persub", **m}


def main() -> int:
    from mwrl_suzuki.run import DEFAULT_CONFIG

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="method-only YAML (fingerprint and MWRL settings)",
    )
    ap.add_argument(
        "--data",
        required=True,
        help="path to the self-contained reaction CSV (see the package README)",
    )
    ap.add_argument("--holdout", type=int, default=13)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="override mwrl.seed (policy + split seed)")
    ap.add_argument("--output", default=None, help="write the table rows to this JSON file")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--arms", choices=("all", "amortized", "persub"), default="all",
                    help="run every arm (default), only the three amortized arms, or only the "
                         "per-substrate ceiling (shardable across jobs)")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="persub shard: this job runs tasks [shard-index::shard-count]")
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    schema = ConditionSchema.from_csv(args.data)
    mcfg = dict(cfg.get("mwrl", {}))
    if args.updates is not None:
        mcfg["updates"] = args.updates
    if args.seed is not None:
        mcfg["seed"] = args.seed
    fp = cfg.get("fingerprint", {})
    seed = int(mcfg.get("seed", 0))
    eval_samples = int(mcfg.get("eval_samples", 200))
    fp_method, n_bits = fp.get("method", "drfp"), int(fp.get("n_bits", 2048))
    per_updates, per_k = int(mcfg.get("per_updates", 200)), int(mcfg.get("per_k", 32))

    # load once here just to compute the split and report; the workers reload in their own processes.
    from mwrl_suzuki.data import load_dataset

    tasks = [t for t in load_dataset(args.data, schema, fingerprint_method=fp_method, n_bits=n_bits)
             if t.antichain]
    ids = [t.task_id for t in tasks]
    order = ids[:]
    random.Random(seed).shuffle(order)
    held_ids, train_ids = order[: args.holdout], order[args.holdout:]
    print(f"[data] {len(tasks)} tasks, {schema.n_dims} dims, "
          f"mean |M|={np.mean([len(t.antichain) for t in tasks]):.2f}; "
          f"{len(train_ids)} train, {len(held_ids)} held-out", flush=True)
    print(f"[parallel] {args.workers} workers over {3 + len(ids)} jobs (3 amortized + {len(ids)} "
          f"per-substrate); credit={mcfg.get('valuation', 'coverage')}/{mcfg.get('baseline', 'l2o')}/"
          f"{mcfg.get('nu', 'geometric')}, k={mcfg.get('k')} updates={mcfg.get('updates')} "
          f"lr={mcfg.get('lr')} horizon={mcfg.get('horizon')} eval={mcfg.get('eval_samples')}\n", flush=True)

    amortized_jobs = [
        {"kind": "amortized", "name": "MWRL", "advantage": "mwrl", "blind": False, "order": 0},
        {"kind": "amortized", "name": "Substrate-blind", "advantage": "mwrl", "blind": True, "order": 1},
        {"kind": "amortized", "name": "Scalar RL", "advantage": "maxrl", "blind": False, "order": 2},
    ] if args.arms in ("all", "amortized") else []
    for j in amortized_jobs:
        j.update({"train_ids": train_ids, "held_ids": held_ids, "mcfg": mcfg, "eval_samples": eval_samples})
    persub_ids = ids[args.shard_index::args.shard_count] if args.arms in ("all", "persub") else []
    persub_jobs = [{"kind": "persub", "task_id": i, "mcfg": mcfg, "per_updates": per_updates,
                    "per_k": per_k, "eval_samples": eval_samples} for i in persub_ids]
    if args.arms != "all":
        print(f"[arms] {args.arms}"
              + (f" shard {args.shard_index}/{args.shard_count} -> {len(persub_ids)} tasks"
                 if args.arms == "persub" else ""), flush=True)

    # Workers only need the tasks their jobs touch. The amortized arms train across the whole
    # library, so they force a full load; a persub-only shard needs just its own slice.
    worker_ids = None if amortized_jobs else tuple(persub_ids)
    del tasks
    gc.collect()   # drop the parent's copy of the grid before the workers spawn

    t_start = time.time()
    amortized_res: list[dict] = []
    persub_res: list[dict] = []
    done_persub = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.data, fp_method, n_bits, seed, worker_ids, schema)) as ex:
        futs = {}
        for j in amortized_jobs:
            futs[ex.submit(_amortized_job, j)] = j
        for j in persub_jobs:
            futs[ex.submit(_persub_job, j)] = j
        for fut in as_completed(futs):
            r = fut.result()
            if r["kind"] == "amortized":
                amortized_res.append(r)
                print(f"[done] {r['name']:<12} recall(held)={r['recall_held']:.2f} "
                      f"#found={r['found']:.2f} born-min={r['born']:.2f} ({r['wall']:.0f}s)", flush=True)
            else:
                persub_res.append(r)
                done_persub += 1
                if done_persub % 8 == 0 or done_persub == len(ids):
                    print(f"[done] per-substrate {done_persub}/{len(ids)}", flush=True)

    # assemble the table.
    rows: list[tuple] = []
    for r in sorted(amortized_res, key=lambda r: r["order"]):
        rows.append((r["name"], r["born"], r["recall_seen"], r["recall_held"], r["found"], r["wall"]))
    if persub_res:
        rows.append(("Per-substrate", _agg(persub_res, "soundness"), _agg(persub_res, "completeness"),
                     None, _agg(persub_res, "n_found"), sum(r["wall"] for r in persub_res)))

    print(f"\n{'Method':<14} {'born-min':>8} {'recall(seen)':>12} {'recall(held)':>12} {'#found':>7} {'train(s)':>9}")
    print("-" * 66)
    for name, born, recall_seen, recall_held, found, wall in rows:
        held_str = f"{recall_held:12.2f}" if recall_held is not None else f"{'---':>12}"
        print(f"{name:<14} {born:8.2f} {recall_seen:12.2f} {held_str} {found:7.2f} {wall:9.0f}", flush=True)
    print(f"\n[wall] whole run finished in {(time.time() - t_start) / 60:.1f} min "
          f"(train(s) column is summed CPU training time per method, not wall-clock)", flush=True)
    if args.output:
        import json
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "seed": seed, "holdout": args.holdout, "mcfg": mcfg,
            "arms": args.arms, "shard": [args.shard_index, args.shard_count],
            "rows": [{"method": n, "born_min": b, "recall_seen": rs, "recall_held": rh,
                      "found": f, "train_s": w} for n, b, rs, rh, f, w in rows],
            # per-task ceiling metrics so persub shards merge exactly across jobs
            "persub_tasks": [{"task_id": r["task_id"], "soundness": r["soundness"],
                              "completeness": r["completeness"], "n_found": r["n_found"],
                              "antichain": r["antichain"], "proposals": r["proposals"],
                              "wall": r["wall"]} for r in persub_res],
            # per-task amortized evaluations (seen + held) for the paired
            # per-substrate figure panels
            "amortized_tasks": {r["name"]: r.get("tasks", [])
                                for r in amortized_res},
        }, indent=2))
        print(f"[saved] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
