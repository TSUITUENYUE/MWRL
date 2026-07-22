"""Task-conditioned sparse-circuit router.

Given a task label, the router activates ONLY that behavior's recovered minimal circuit (the
rest of the model resample-ablated) and shows the model still does the task, close to the full
dense model, on ~10% of the components. The confusion matrix -- task i's prompts run on circuit
j -- shows the circuits are task-specific: the diagonal (correct routing) tracks the dense model,
the off-diagonal (wrong circuit) collapses. This is the MoE-ification claim: one dense model, a
per-task sparse circuit selected by the label, near-dense accuracy at a fraction of the compute.

    python -m mwrl_circuits.router --config configs/qwen3_1p7b.yaml --circuits runs/circuits/mwrl/report.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import yaml

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.tasks import build_hierarchy
from mwrl_circuits.verifier import CircuitVerifier


def _name_to_idx(name: str, model: AblatedModel) -> int:
    """Inverse of run._component_name: 'a{layer}.h{head}' -> head index, 'mlp{layer}' -> mlp index."""
    if name.startswith("a") and ".h" in name:
        layer, head = name[1:].split(".h")
        return int(layer) * model.n_heads + int(head)
    return model.n_attn + int(name[len("mlp"):])


def _mask_from_names(names: list[str], model: AblatedModel) -> int:
    mask = 0
    for nm in names:
        mask |= 1 << _name_to_idx(nm, model)
    return mask


@torch.no_grad()
def _stats(verifier: CircuitVerifier, keep_mask: int) -> tuple[float, float]:
    """(accuracy = fraction of prompts with correct > foil, mean correct-minus-foil margin)
    running the behavior's prompts with only ``keep_mask`` active (rest resample-ablated)."""
    keep = None if keep_mask == (1 << verifier.n_comp) - 1 else verifier._keep(keep_mask)
    logits = verifier.model.logits(verifier.enc, keep, verifier.cf)
    diff = logits[verifier.rows, verifier.correct] - logits[verifier.rows, verifier.foil]
    return float((diff > 0).float().mean()), float(diff.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--circuits", required=True, help="an mwrl report.json with recovered circuits")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    report = json.load(open(args.circuits))
    tasks = [s for node in build_hierarchy(cfg["families"]).values() for s in node["subtasks"]]

    # one representative circuit per behavior: the smallest recovered minimal circuit.
    verifiers: dict[str, CircuitVerifier] = {}
    circuits: dict[str, int] = {}
    for t in tasks:
        entry = report.get(t.name)
        if not entry or not entry.get("circuits"):
            continue
        verifiers[t.name] = CircuitVerifier(model, t, tau=cfg["tau"])
        circuits[t.name] = _mask_from_names(min(entry["circuits"], key=len), model)
    names = list(verifiers)
    full = (1 << model.n_comp) - 1
    print(f"[router] {len(names)} behaviors, n_comp={model.n_comp}\n", flush=True)

    # confusion matrix of accuracy: row = task i's prompts, col = circuit j.
    header = "acc: task_i on circuit_j"
    print(f"{header:>26s} | " + " ".join(f"{n[:7]:>7s}" for n in names))
    diag_acc, diag_ret, sizes = [], [], []
    for i in names:
        v = verifiers[i]
        full_acc, full_margin = _stats(v, full)
        row = []
        for j in names:
            acc, _ = _stats(v, circuits[j])
            row.append(acc)
        own_acc, own_margin = _stats(v, circuits[i])
        size = bin(circuits[i]).count("1")
        diag_acc.append(own_acc)
        diag_ret.append(own_margin / full_margin if full_margin else 0.0)
        sizes.append(size)
        print(f"{i:16s} full={full_acc:.2f} | " + " ".join(f"{a:7.2f}" for a in row))

    off = [_stats(verifiers[i], circuits[j])[0] for i in names for j in names if i != j]
    print("\n[summary]")
    print(f"  own-circuit accuracy (router, diagonal):   {np.mean(diag_acc):.2f}  (dense = ~1.0)")
    print(f"  wrong-circuit accuracy (off-diagonal):     {np.mean(off):.2f}")
    print(f"  margin retained by own circuit:            {np.mean(diag_ret):.0%}")
    print(f"  components active per task:                 {np.mean(sizes):.0f}/{model.n_comp}"
          f"  ({np.mean(sizes) / model.n_comp:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
