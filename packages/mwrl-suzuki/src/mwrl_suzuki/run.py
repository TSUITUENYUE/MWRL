"""Recover minimal condition-set antichains from the generated Suzuki CSV.

    python -m mwrl_suzuki.run

One policy is trained over the training substrate pairs, conditioned on each pair's reaction
fingerprint, and evaluated on every pair including a held-out split it never trained on. The
CSV supplies the complete dataset schema; the optional YAML contains method settings only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml
from mwrl.config import MWRLConfig
from mwrl.runner import Runner

from mwrl_suzuki.data import load_dataset
from mwrl_suzuki.env import ChemEnv
from mwrl_suzuki.evaluate import evaluate_antichain
from mwrl_suzuki.fingerprint import drfp_available
from mwrl_suzuki.schema import ConditionSchema
from mwrl_suzuki.verifier import ChemVerifier

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "method.yaml"


def build_config(mwrl_cfg: dict) -> MWRLConfig:
    # The validated coverage credit contract: coverage valuation with the exact leave-two-out
    # centering (l2o, unbiased despite the U_K coupling), the geometric up-set measure (p=0.7),
    # and mean-absolute scale. Same recipe as the circuits/MaxSAT coverage runs.
    return MWRLConfig(
        witness_valuation=mwrl_cfg.get("valuation", "coverage"),
        witness_center=True,
        witness_baseline=mwrl_cfg.get("baseline", "l2o"),
        witness_normalize=True,
        witness_scale="mean",
        witness_nu=mwrl_cfg.get("nu", "geometric"),
        witness_nu_p=mwrl_cfg.get("nu_p", 0.7),
        witness_mc_samples=mwrl_cfg.get("mc_samples", 20000),
        updates=mwrl_cfg.get("updates", 200),
        task_batch_size=mwrl_cfg.get("groups_per_update", 8),
        group_size=mwrl_cfg.get("k", 12),
        learning_rate=mwrl_cfg.get("lr", 3e-3),
        entropy_coef=mwrl_cfg.get("entropy_coef", 0.02),
        hidden_dim=mwrl_cfg.get("hidden", 256),
        update_epochs=1,
        minibatch_size=256,
        clip_epsilon=0.2,
        max_grad_norm=1.0,
        epsilon=1e-6,
    )


def _component_names(mask: int, schema: ConditionSchema) -> list[str]:
    return [schema.dimensions[i] for i in range(schema.n_dims) if (mask >> i) & 1]


def main() -> int:
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
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="fraction of substrate pairs held out")
    ap.add_argument("--updates", type=int, default=None, help="override mwrl.updates")
    ap.add_argument("--seed", type=int, default=None, help="override mwrl.seed (policy + split seed)")
    ap.add_argument("--output", default="runs/suzuki/latest")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    schema = ConditionSchema.from_csv(args.data)
    mcfg = dict(cfg.get("mwrl", {}))
    if args.updates is not None:
        mcfg["updates"] = args.updates
    if args.seed is not None:
        mcfg["seed"] = args.seed
    fp_cfg = cfg.get("fingerprint", {})
    seed = int(mcfg.get("seed", 0))
    eval_samples = int(mcfg.get("eval_samples", 128))

    fp_method = fp_cfg.get("method", "drfp")
    if fp_method == "drfp" and not drfp_available():
        print("[error] fingerprint method 'drfp' requires the drfp package; install with "
              "pip install 'mwrl-suzuki[fp]', or set method: descriptors in the config.", flush=True)
        return 1
    fp_label = fp_method

    tasks = load_dataset(
        args.data, schema, fingerprint_method=fp_method, n_bits=int(fp_cfg.get("n_bits", 2048)),
    )
    tasks = [t for t in tasks if t.antichain]  # drop tasks with no measured success (empty antichain)
    if not tasks:
        print("[error] no task has a measured success at the threshold", flush=True)
        return 1
    print(f"[data] {len(tasks)} tasks, {schema.n_dims} dimensions, "
          f"fingerprint={fp_label} dim={len(tasks[0].fingerprint)}", flush=True)

    rng = random.Random(seed)
    order = list(range(len(tasks)))
    rng.shuffle(order)
    n_held = max(1, int(round(args.holdout_frac * len(tasks)))) if args.holdout_frac > 0 else 0
    held_ids = {tasks[i].task_id for i in order[:n_held]}
    verifiers = {t.task_id: ChemVerifier(t) for t in tasks}
    train_v = [verifiers[t.task_id] for t in tasks if t.task_id not in held_ids]
    if not train_v:
        print("[error] holdout fraction left no training tasks", flush=True)
        return 1
    fp_dim = len(train_v[0].attr_features)
    print(f"[amortize] one policy over {len(train_v)} train tasks, {len(held_ids)} held out", flush=True)

    env = ChemEnv(train_v, n_comp=schema.n_dims, fp_dim=fp_dim,
                  horizon=int(mcfg.get("horizon", schema.n_dims)))
    config = build_config(mcfg)
    print(f"[resolved-config] valuation={config.witness_valuation} baseline={config.witness_baseline} "
          f"scale={config.witness_scale} nu={config.witness_nu} nu_p={config.witness_nu_p} "
          f"k={config.group_size} groups={config.task_batch_size} updates={config.updates} "
          f"lr={config.learning_rate} horizon={env.horizon}", flush=True)
    runner = Runner(env, config, advantage="mwrl", optimizer="reinforce", seed=seed)
    t0 = time.time()

    def progress(u: int, sr: float) -> None:
        if u % max(1, config.updates // 10) == 0 or u == config.updates - 1:
            print(f"[update {u + 1}/{config.updates}] success={sr:.2f} {time.time() - t0:.0f}s", flush=True)

    runner.train(on_update=progress)

    report: dict = {}
    for t in tasks:
        v = verifiers[t.task_id]
        m = evaluate_antichain(runner, v, n_samples=eval_samples)
        m["heldout"] = t.task_id in held_ids
        m["antichain"] = [_component_names(w, schema) for w in v.ground_truth_antichain()]
        report[t.task_id] = m

    def _agg(key: str, held: bool) -> float:
        vals = [m[key] for m in report.values() if isinstance(m, dict) and m.get("heldout") == held]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'task':16s} {'split':>8s} {'|M|':>4s} {'sound':>6s} {'compl':>6s} {'#found':>7s} {'faith':>6s}")
    print("-" * 62)
    for name, m in report.items():
        split = "heldout" if m["heldout"] else "train"
        print(f"{name:16s} {split:>8s} {m['antichain_size']:4d} {m['soundness']:6.2f} "
              f"{m['completeness']:6.2f} {m['n_found']:7d} {m['mean_faithfulness']:6.2f}")
    print("-" * 62)
    for held, label in [(False, "train"), (True, "heldout")]:
        print(f"{'MEAN':16s} {label:>8s} {'':>4s} {_agg('soundness', held):6.2f} "
              f"{_agg('completeness', held):6.2f} {_agg('n_found', held):7.1f} {_agg('mean_faithfulness', held):6.2f}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report["_meta"] = {"n_tasks": len(tasks), "n_dims": schema.n_dims, "held_out": sorted(held_ids),
                       "seconds": round(time.time() - t0, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] {out / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
