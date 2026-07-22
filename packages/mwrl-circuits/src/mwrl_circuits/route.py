"""Predictive task router over the 3-level hash table: prompt -> leaf -> activate that leaf's circuit.

The oracle router (``router.py``) shows the circuits are task-specific given the label. This closes
the loop: a cheap classifier on the model's own input embeddings predicts a prompt's family /
subtask / leaf, and only the predicted leaf's sparse circuit is activated (the rest resample-
ablated). We report

  * routing accuracy at each level (family / subtask / leaf) on held-out prompts;
  * the sparse-circuit performance end-to-end -- each held-out prompt run under the ROUTER-selected
    circuit -- against the dense model and against oracle routing (the true leaf's circuit);
  * the MoE-ification numbers: mean components active per task and the accuracy retained.

The router features are the mean input-token embedding (an embedding lookup + a linear head, far
cheaper than a dense forward), so the story is: cheap routing + a per-leaf sparse circuit ~ dense.

    python -m mwrl_circuits.route --config configs/qwen3_8b.yaml --hashtable runs/circuits/hier3/hashtable.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.tasks import build_taxonomy
from mwrl_circuits.verifier import CircuitVerifier


@torch.no_grad()
def _mean_embedding(model: AblatedModel, enc: dict) -> torch.Tensor:
    """Mean input-token embedding over the real (non-pad) tokens of each prompt: [B, d]. This is
    the cheap router feature -- an embedding lookup, no transformer forward."""
    emb = model.model.get_input_embeddings()(enc["input_ids"])       # [B, L, d]
    mask = enc["attention_mask"].unsqueeze(-1).to(emb.dtype)          # [B, L, 1]
    return ((emb * mask).sum(1) / mask.sum(1).clamp(min=1)).float()   # [B, d]


def _slice(v: CircuitVerifier, idx: list[int]):
    """The leaf's prompts ``idx`` as (enc, cf) both sliced to those rows (the corrupt activations
    must match the enc rows or the ablation ``torch.where`` sees mismatched batch sizes)."""
    sel = torch.as_tensor(idx, device=v.model.device)
    enc = {k: val[sel] for k, val in v.enc.items()}
    cf_attn, cf_mlp = v.cf
    return enc, ([a[sel] for a in cf_attn], [m[sel] for m in cf_mlp]), sel


@torch.no_grad()
def _kl_faith(v: CircuitVerifier, idx: list[int], keep: torch.Tensor | None) -> float:
    """Mean distributional faithfulness 1 - KL(P_clean || P_keep)/KL_empty over prompts ``idx`` --
    the SAME measure the verifier's s0 thresholds. ``keep`` is None (dense = 1.0), one [n_comp] mask,
    or a per-prompt [len(idx), n_comp] mask (router). Computed per prompt so per-prompt masks work."""
    if keep is None or v.kl_empty <= 0:
        return 1.0
    enc, cf, sel = _slice(v, idx)
    logp_c = torch.log_softmax(v.clean_logits[sel].float(), dim=-1)
    p_c = logp_c.exp()
    logp_s = torch.log_softmax(v.model.logits(enc, keep, cf).float(), dim=-1)
    kl = (p_c * (logp_c - logp_s)).sum(-1)                       # [t]
    return float((1.0 - kl / v.kl_empty).mean())


@torch.no_grad()
def _gen_match(v: CircuitVerifier, idx: list[int], keep: torch.Tensor | None, golds: list[str]) -> int:
    """Greedy-decode the answer under ``keep`` and count prompts whose generation begins with the
    gold answer string -- real task performance (does the sparse model produce the answer), not a
    two-way margin. ``golds`` is the gold string per prompt in ``idx`` (aligned)."""
    enc, cf, _ = _slice(v, idx)
    gen = v.model.generate(enc, keep, cf, max_new_tokens=4)      # [t, 4] token ids
    return sum(v.model.tokenizer.decode(gen[k]).strip().startswith(golds[k].strip())
               for k in range(len(idx)))


def _split(n: int, frac: float, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """Per-leaf train/test split so every leaf appears in both (at least one test prompt)."""
    perm = rng.permutation(n)
    n_tr = max(1, min(n - 1, round(frac * n)))
    return sorted(perm[:n_tr].tolist()), sorted(perm[n_tr:].tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hashtable", required=True, help="hier3 hashtable.json with leaf_masks")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    ht = json.load(open(args.hashtable))
    leaf_masks: dict[str, int] = {k: int(v) for k, v in ht["leaf_masks"].items()}
    if not leaf_masks:
        print("[abort] hashtable has no leaf_masks (the discovery run recovered no leaf circuit)")
        return 1

    # -- assemble the leaves that have a recovered circuit, with their tree labels ----------------
    tax = build_taxonomy()
    leaves, verifiers, masks, fam_of, sub_of, golds = [], {}, {}, {}, {}, {}
    for fam, node in tax.items():
        for sub, snode in node["subtasks"].items():
            for lf in snode["leaves"]:
                key = f"{fam}/{sub}/{lf.name}"
                if key not in leaf_masks:
                    continue
                leaves.append(lf.name)
                verifiers[lf.name] = CircuitVerifier(model, lf, tau=cfg["tau"])
                masks[lf.name] = leaf_masks[key]
                fam_of[lf.name], sub_of[lf.name] = fam, f"{fam}/{sub}"
                golds[lf.name] = list(lf.correct)               # gold answer string per prompt
    leaf_idx = {name: i for i, name in enumerate(leaves)}
    K, rng = len(leaves), np.random.default_rng(args.seed)
    print(f"[router] {K} leaves with circuits, n_comp={model.n_comp}\n", flush=True)

    # -- features + per-leaf train/test split ----------------------------------------------------
    Xtr, ytr, test = [], [], {}     # test[name] = held-out prompt indices for that leaf
    for name in leaves:
        v = verifiers[name]
        feats = _mean_embedding(model, v.enc)                        # [n, d]
        tr, te = _split(feats.shape[0], args.train_frac, rng)
        Xtr.append(feats[tr]); ytr += [leaf_idx[name]] * len(tr)
        test[name] = te
    Xtr = torch.cat(Xtr); ytr = torch.tensor(ytr, device=device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6

    # -- train the linear router (leaf classification on standardized embeddings) ----------------
    torch.manual_seed(args.seed)
    clf = nn.Linear(Xtr.shape[1], K).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=0.02, weight_decay=1e-3)
    Xn = (Xtr - mu) / sd
    for _ in range(400):
        opt.zero_grad(); F.cross_entropy(clf(Xn), ytr).backward(); opt.step()

    # -- evaluate: routing accuracy + distributional faithfulness + real generation performance ---
    r_leaf = r_sub = r_fam = n_test = 0
    gen_dense = gen_oracle = gen_router = 0
    faith_oracle = faith_router = 0.0
    sizes = []
    for name in leaves:
        v, te = verifiers[name], test[name]
        with torch.no_grad():
            pred = clf(((_mean_embedding(model, v.enc)[te]) - mu) / sd).argmax(1).tolist()
        pred_names = [leaves[p] for p in pred]
        r_leaf += sum(pn == name for pn in pred_names)
        r_sub += sum(sub_of[pn] == sub_of[name] for pn in pred_names)
        r_fam += sum(fam_of[pn] == fam_of[name] for pn in pred_names)
        n_test += len(te)
        sizes.append(bin(masks[name]).count("1"))
        te_golds = [golds[name][i] for i in te]
        keep_oracle = v._keep(masks[name])                                    # [n_comp]
        keep_router = torch.stack([verifiers[pn]._keep(masks[pn]) for pn in pred_names])  # [t, n_comp]
        # real task performance: greedy-generate the answer under each routing, exact-match the gold
        gen_dense += _gen_match(v, te, None, te_golds)
        gen_oracle += _gen_match(v, te, keep_oracle, te_golds)
        gen_router += _gen_match(v, te, keep_router, te_golds)
        # distributional faithfulness (the verifier's own KL measure), prompt-weighted
        faith_oracle += _kl_faith(v, te, keep_oracle) * len(te)
        faith_router += _kl_faith(v, te, keep_router) * len(te)

    print(f"[routing accuracy on {n_test} held-out prompts]")
    print(f"  family / subtask / leaf:         {r_fam / n_test:.2f} / {r_sub / n_test:.2f} / "
          f"{r_leaf / n_test:.2f}")
    print("\n[distributional faithfulness  1 - KL(P_clean || P_circuit) / KL_empty]")
    print(f"  oracle-routed (true circuit):    {faith_oracle / n_test:.3f}")
    print(f"  router-routed (predicted):       {faith_router / n_test:.3f}   (dense = 1.000)")
    print("\n[real task performance  (greedy-generation exact match)]")
    print(f"  dense model:                     {gen_dense / n_test:.2f}")
    print(f"  oracle-routed (true circuit):    {gen_oracle / n_test:.2f}"
          f"   (retains {gen_oracle / max(gen_dense, 1):.0%})")
    print(f"  router-routed (predicted):       {gen_router / n_test:.2f}"
          f"   (retains {gen_router / max(gen_dense, 1):.0%})")
    print(f"\n  components active per task:       {np.mean(sizes):.0f}/{model.n_comp}"
          f"  ({np.mean(sizes) / model.n_comp:.0%})")

    # -- keep the discovery metrics (soundness / n_found / faithfulness / size) -------------------
    reports = ht.get("reports", {}).get("subsubtask", {})
    if reports:
        print(f"\n[discovery metrics per leaf]  {'leaf':16s} {'#found':>7s} {'faith':>6s} {'size':>6s}")
        for name in leaves:
            r = reports.get(name)
            if r:
                print(f"  {name:16s} {r['n_found']:7d} {r.get('mean_faithfulness', 0):6.2f} "
                      f"{r['median_size']:6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
