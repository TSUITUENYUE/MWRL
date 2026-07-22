"""Dense-model validation of the taxonomy leaves.

Before an expensive discovery run commits to the 24-leaf taxonomy, check which leaves the target
model actually does: each leaf's full-model correct-minus-foil margin and its accuracy (fraction of
prompts with correct > foil). A leaf passes if the margin is positive -- the discovery pipeline
skips the rest, so this is just the cheap up-front view of what will survive. Low accuracy on a
positive-margin leaf flags prompts/foils worth fixing before the run.

    python -m mwrl_circuits.validate --config configs/qwen3_8b.yaml --output runs/circuits/leaves.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import yaml

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.tasks import build_taxonomy
from mwrl_circuits.verifier import CircuitVerifier


@torch.no_grad()
def _dense_accuracy(v: CircuitVerifier) -> float:
    """Fraction of the leaf's prompts the full (unablated) model gets right (correct > foil)."""
    logits = v.model.logits(v.enc, None, v.cf)
    diff = logits[v.rows, v.correct] - logits[v.rows, v.foil]
    return float((diff > 0).float().mean())


def validate_leaves(model: AblatedModel, tau: float) -> list[dict]:
    """Per-leaf dense margin + accuracy over the whole taxonomy, in tree order."""
    rows: list[dict] = []
    for fam, node in build_taxonomy().items():
        for sub, snode in node["subtasks"].items():
            for lf in snode["leaves"]:
                v = CircuitVerifier(model, lf, tau=tau)
                rows.append({"family": fam, "subtask": sub, "leaf": lf.name,
                             "kl_signal": v.kl_empty, "margin": v.full_margin,
                             "accuracy": _dense_accuracy(v), "n": len(lf.prompts),
                             "pass": v.full_margin > 1e-6})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    rows = validate_leaves(model, cfg["tau"])
    print(f"\n{'family':11s} {'subtask':11s} {'leaf':16s} {'KL_sig':>7s} {'margin':>7s} "
          f"{'acc':>5s} {'n':>3s}  ok")
    for r in rows:
        print(f"{r['family']:11s} {r['subtask']:11s} {r['leaf']:16s} {r['kl_signal']:7.2f} "
              f"{r['margin']:7.2f} {r['accuracy']:5.2f} {r['n']:3d}  {'Y' if r['pass'] else '.'}")
    npass = sum(r["pass"] for r in rows)
    hiacc = sum(r["accuracy"] >= 0.75 for r in rows)
    print(f"\n[summary] {npass}/{len(rows)} leaves pass (margin>0); "
          f"{hiacc}/{len(rows)} at acc>=0.75.  KL_sig = interchange signal the circuit must recover",
          flush=True)
    if args.output:
        json.dump(rows, open(args.output, "w"), indent=2)
        print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
