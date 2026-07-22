"""Hierarchical MoE evaluation: the leaf-by-circuit confusion matrix and the
active-parameter table (per-task sparse expert vs the dense model).

Rows are every taxonomy leaf the dense model can do (margin accuracy at the answer
position); columns are every leaf with a recovered circuit in the hashtable. Cell (i, j)
is task i's accuracy with ONLY circuit j active (complement resample-ablated). The
parameter table converts each circuit's component set into exact active parameters from
the model config: a kept head owns its q and o slices plus its GQA share of k and v; a
kept MLP block owns gate, up, and down; embeddings, norms, and the lm_head are the
always-on backbone counted in every routed forward.

    python -m mwrl_circuits.hier_moe_eval --config configs/qwen3_8b_hcov.yaml \
        --hashtable runs/circuits/mwrl-hcov8b_476625/hashtable.json --output runs/.../moe_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import yaml

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.router import _stats
from mwrl_circuits.tasks import build_taxonomy
from mwrl_circuits.verifier import CircuitVerifier


def component_params(model: AblatedModel) -> tuple[list[int], int, int]:
    """Exact parameter count per ablatable component, the always-on backbone count, and
    the dense total. Heads carry q+o plus their GQA share of k+v; MLPs carry gate+up+down."""
    cfg = model.model.config
    hidden, head_dim = cfg.hidden_size, model.head_dim
    n_kv = int(getattr(cfg, "num_key_value_heads", model.n_heads))
    share = model.n_heads // max(1, n_kv)                  # query heads per kv group
    per_head = hidden * head_dim * 2 + (hidden * head_dim * 2) // share   # q,o + kv share
    inter = cfg.intermediate_size
    per_mlp = 3 * hidden * inter                            # gate, up, down
    counts = [per_head] * model.n_attn + [per_mlp] * model.n_layers
    total = sum(p.numel() for p in model.model.parameters())
    backbone = total - sum(counts)
    return counts, backbone, total


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hashtable", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg = cfg.get("mwrl", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    masks = {k: int(v) for k, v in json.load(open(args.hashtable))["leaf_masks"].items()}
    circuits = {k: v for k, v in masks.items() if v}
    counts, backbone, total = component_params(model)
    print(f"[moe] {len(circuits)} leaf circuits, n_comp={model.n_comp}, "
          f"dense params={total/1e9:.2f}B (backbone {backbone/1e9:.2f}B)", flush=True)

    leaves = {f"{fam}/{sub}/{lf.name}": lf
              for fam, node in build_taxonomy().items()
              for sub, snode in node["subtasks"].items() for lf in snode["leaves"]}
    batch_seqs = int(mcfg.get("batch_seqs", 512))
    rows: list[str] = []
    verifiers: dict[str, CircuitVerifier] = {}
    dense_acc: dict[str, float] = {}
    for path, leaf in leaves.items():
        v = CircuitVerifier(model, leaf, tau=cfg["tau"], batch_seqs=batch_seqs)
        acc, _ = _stats(v, (1 << model.n_comp) - 1)
        if v.full_metric <= 1e-6 or acc <= 0.5:
            print(f"[skip-row] {path}: dense acc {acc:.2f}", flush=True)
            continue
        rows.append(path)
        verifiers[path], dense_acc[path] = v, acc

    cols = [p for p in circuits if p in leaves]
    matrix = [[_stats(verifiers[r], circuits[c])[0] for c in cols] for r in rows]
    for r, line in zip(rows, matrix, strict=True):
        print(f"{r:44s} dense={dense_acc[r]:.2f} | " +
              " ".join(f"{x:4.2f}" for x in line), flush=True)

    table: dict[str, dict] = {}
    for c in cols:
        size = bin(circuits[c]).count("1")
        active = backbone + sum(counts[i] for i in range(model.n_comp) if (circuits[c] >> i) & 1)
        own = matrix[rows.index(c)][cols.index(c)] if c in rows else None
        table[c] = {"size": size, "active_params": active, "active_frac": active / total,
                    "dense_acc": dense_acc.get(c), "own_acc": own,
                    "retention": (own / dense_acc[c]) if (own is not None and dense_acc.get(c)) else None}
        print(f"[expert] {c:44s} comps={size:4d} active={active/1e9:.2f}B "
              f"({active/total:5.1%}) own={own if own is None else f'{own:.2f}'} "
              f"dense={dense_acc.get(c) and f'{dense_acc[c]:.2f}'}", flush=True)

    diag = [table[c]["own_acc"] for c in cols if table[c]["own_acc"] is not None]
    off = [matrix[i][j] for i, r in enumerate(rows) for j, c in enumerate(cols)
           if r != c and table[c]["own_acc"] is not None]
    summary = {"n_rows": len(rows), "n_experts": len(cols),
               "mean_active_frac": sum(table[c]["active_frac"] for c in cols) / max(1, len(cols)),
               "mean_retention": sum(table[c]["retention"] for c in cols
                                     if table[c]["retention"] is not None)
                                 / max(1, sum(1 for c in cols if table[c]["retention"] is not None)),
               "confusion_diag": sum(diag) / max(1, len(diag)),
               "confusion_off": sum(off) / max(1, len(off)),
               "dense_params": total, "backbone_params": backbone}
    print("\n[summary] " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in summary.items()}), flush=True)
    json.dump({"rows": rows, "cols": cols, "matrix": matrix, "dense_acc": dense_acc,
               "experts": table, "summary": summary}, open(args.output, "w"), indent=1)
    print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
