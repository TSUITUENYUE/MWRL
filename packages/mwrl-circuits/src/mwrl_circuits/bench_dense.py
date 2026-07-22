"""Dense-model reference accuracy on the benchmark families.

The dense model's own accuracy on each subject is the reference every retention number is
measured against, and it selects the expert set: sweep all 57 MMLU subjects, keep every
subject above the bar, discover circuits on all that pass. One unablated forward per
subject.

    python -m mwrl_circuits.bench_dense --config configs/qwen3_1p7b_bench.yaml \
        --subjects all --bar 0.75 --output runs/bench_dense_1p7b.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import yaml

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.tasks import build_hierarchy
from mwrl_circuits.verifier import CircuitVerifier


@torch.no_grad()
def _dense(model: AblatedModel, task, tau: float) -> tuple[float, float]:
    v = CircuitVerifier(model, task, tau=tau)
    logits = v.model.logits(v.enc, None, v.cf)
    diff = logits[v.rows, v.correct] - logits[v.rows, v.foil]
    return float((diff > 0).float().mean()), float(diff.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subjects", default="config",
                    help='"config" = the config families; "all" = every MMLU subject; '
                         'or a comma-separated subject list')
    ap.add_argument("--bar", type=float, default=0.75, help="dense-accuracy pass bar")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))

    if args.subjects == "config":
        tasks = [t for node in build_hierarchy(cfg["families"]).values()
                 for t in node["subtasks"]]
    else:
        from mwrl_circuits.bench import all_mmlu_subjects, mmlu_task
        names = all_mmlu_subjects() if args.subjects == "all" else args.subjects.split(",")
        tasks, skipped = [], []
        for s in names:
            try:
                tasks.append(mmlu_task(s))
            except ValueError as e:
                skipped.append(s)
                print(f"[skip] {e}", flush=True)
        if skipped:
            print(f"[skip] {len(skipped)} subjects lacked short balanced questions", flush=True)

    out: dict[str, dict] = {}
    for task in tasks:
        acc, margin = _dense(model, task, cfg["tau"])
        out[task.name] = {"dense_acc": acc, "dense_margin": margin, "n": len(task.prompts)}
        print(f"{task.name:42s} dense_acc={acc:.2f}  margin={margin:+.3f}  n={len(task.prompts)}",
              flush=True)

    ranked = sorted(out.items(), key=lambda kv: -kv[1]["dense_acc"])
    passing = [name for name, r in ranked if r["dense_acc"] >= args.bar]
    print(f"\n[pass >= {args.bar:.2f}] {len(passing)}/{len(out)} subjects", flush=True)
    for name in passing:
        print(f"  {name}  {out[name]['dense_acc']:.2f}", flush=True)
    if args.output:
        json.dump({"bar": args.bar, "passing": passing, "results": out},
                  open(args.output, "w"), indent=1)
        print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
