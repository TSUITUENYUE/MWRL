"""Benchmark retention table + confusion matrix, run in the same job as discovery.

For every behavior with a recovered circuit, evaluate at MATCHED sparsity (the size of the
behavior's smallest recovered circuit):

  dense      the unablated model (the retention reference)
  mwrl       the recovered circuit itself
  eap        the top-k components by |EAP attribution| (the ACDC-style baseline)
  magnitude  the top-k components by mean |weight| of their output projection
  random     a uniformly sampled k-subset (seeded), the sanity floor

plus the full task-by-circuit confusion matrix (own-circuit diagonal vs off-diagonal).
Accuracy is the codebase's pairwise margin (correct token beats the cyclic foil token) with
the complement resample-ablated, identical to router.py.

    python -m mwrl_circuits.bench_eval --config configs/qwen3_1p7b_bench.yaml \
        --circuits runs/.../report.json --output runs/.../bench_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import torch
import yaml

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.baselines_masks import acdc_mask, hardconcrete_mask, wanda_scores
from mwrl_circuits.router import _mask_from_names, _stats
from mwrl_circuits.tasks import build_hierarchy
from mwrl_circuits.verifier import CircuitVerifier


def _magnitude_scores(model: AblatedModel) -> torch.Tensor:
    """Mean |weight| of each component's output projection (per-element, so heads and MLP
    blocks are comparable): heads score their o_proj column block, MLPs their down_proj."""
    scores = torch.zeros(model.n_comp)
    for layer_idx, layer in enumerate(model._layers):
        o_w = layer.self_attn.o_proj.weight
        for h in range(model.n_heads):
            block = o_w[:, h * model.head_dim:(h + 1) * model.head_dim]
            scores[layer_idx * model.n_heads + h] = block.abs().mean()
        scores[model.n_attn + layer_idx] = layer.mlp.down_proj.weight.abs().mean()
    return scores


def _topk_mask(scores: torch.Tensor, k: int) -> int:
    mask = 0
    for i in torch.topk(scores, k).indices.tolist():
        mask |= 1 << i
    return mask


def _random_mask(n_comp: int, k: int, seed: int) -> int:
    mask = 0
    for i in random.Random(seed).sample(range(n_comp), k):
        mask |= 1 << i
    return mask


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--circuits", required=True, help="report.json from the discovery run")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    report = json.load(open(args.circuits))
    tasks = [t for node in build_hierarchy(cfg["families"]).values() for t in node["subtasks"]]
    magnitude = _magnitude_scores(model)

    verifiers: dict[str, CircuitVerifier] = {}
    circuits: dict[str, int] = {}
    for t in tasks:
        entry = report.get(t.name)
        if not entry or not entry.get("circuits") or not min(entry["circuits"], key=len):
            print(f"[skip] {t.name}: no recovered circuit", flush=True)
            continue
        verifiers[t.name] = CircuitVerifier(model, t, tau=cfg["tau"],
                                            batch_seqs=int(cfg.get("mwrl", {}).get("batch_seqs", 1024)))
        circuits[t.name] = _mask_from_names(min(entry["circuits"], key=len), model)
    names = list(verifiers)
    print(f"[bench_eval] {len(names)} behaviors, n_comp={model.n_comp}\n", flush=True)

    header = (f"{'behavior':30s} {'size':>4s} {'dense':>6s} {'mwrl':>6s} {'eap':>6s} {'magn':>6s}"
              f" {'rand':>6s} {'wanda':>6s} {'hc':>6s} {'acdc':>6s} {'aSz':>4s}")
    print(header, flush=True)
    rows: dict[str, dict] = {}
    for i, name in enumerate(names):
        v, own = verifiers[name], circuits[name]
        k = bin(own).count("1")
        acdc = acdc_mask(v)
        r = {"size": k,
             "dense": _stats(v, (1 << model.n_comp) - 1)[0],
             "mwrl": _stats(v, own)[0],
             "eap": _stats(v, _topk_mask(v.attrs.abs().cpu(), k))[0],
             "magnitude": _stats(v, _topk_mask(magnitude, k))[0],
             "random": _stats(v, _random_mask(model.n_comp, k, args.seed + i))[0],
             "wanda": _stats(v, _topk_mask(wanda_scores(model, v), k))[0],
             "hardconcrete": _stats(v, hardconcrete_mask(model, v, k, seed=args.seed))[0],
             "acdc": _stats(v, acdc)[0], "acdc_size": bin(acdc).count("1")}
        rows[name] = r
        print(f"{name:30s} {k:4d} {r['dense']:6.2f} {r['mwrl']:6.2f} {r['eap']:6.2f}"
              f" {r['magnitude']:6.2f} {r['random']:6.2f} {r['wanda']:6.2f}"
              f" {r['hardconcrete']:6.2f} {r['acdc']:6.2f} {r['acdc_size']:4d}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[confusion] task_i prompts on circuit_j", flush=True)
    confusion = [[_stats(verifiers[i], circuits[j])[0] for j in names] for i in names]
    diag = [confusion[i][i] for i in range(len(names))]
    off = [confusion[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    summary = {n: mean([rows[b][n] for b in rows])
               for n in ("dense", "mwrl", "eap", "magnitude", "random", "wanda",
                         "hardconcrete", "acdc")}
    summary.update({"size": mean([rows[b]["size"] for b in rows]),
                    "active_frac": mean([rows[b]["size"] for b in rows]) / model.n_comp,
                    "confusion_diag": mean(diag), "confusion_off": mean(off)})
    print("\n[summary]", flush=True)
    for key, val in summary.items():
        print(f"  {key:16s} {val:.3f}", flush=True)
    json.dump({"rows": rows, "confusion": {"names": names, "matrix": confusion},
               "summary": summary}, open(args.output, "w"), indent=1)
    print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
