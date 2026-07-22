"""Three-level hierarchical MWRL: family -> subtask -> subsubtask (the hash-table taxonomy).

Each level is one amortized MWRL pass, restricted to its parent's recovered support and warm-started
from the parent policy (reuse across granularity = the amortization claim). Output is the nested
hash table {family: {subtask: {subsubtask: circuit_size}}} plus each level's sparse-circuit eval.
No router here -- routing is the L40/8B step; this verifies the hierarchical idea at 4B.

    python -m mwrl_circuits.hierarchical3 --config configs/qwen3_4b.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.hierarchical import _amortized, _support
from mwrl_circuits.tasks import build_taxonomy
from mwrl_circuits.verifier import CircuitVerifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="runs/circuits/hier3")
    ap.add_argument("--updates", type=int, default=None, help="override mwrl.updates (all passes)")
    ap.add_argument("--eval-samples", type=int, default=None,
                    help="override mwrl.eval_samples (the per-behavior support/antichain samples)")
    ap.add_argument("--batch-seqs", type=int, default=None,
                    help="override mwrl.batch_seqs (the masks*prompts sequence budget per forward; "
                         "set to the memory budget -- batch_masks auto-scales as budget/prompts, so "
                         "the coarse 60-prompt family passes stop underfilling the GPU)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg = dict(cfg.get("mwrl", {}))
    if args.updates is not None:
        mcfg["updates"] = args.updates
    if args.eval_samples is not None:
        mcfg["eval_samples"] = args.eval_samples
    if args.batch_seqs is not None:
        mcfg["batch_seqs"] = args.batch_seqs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    tau, seed = cfg["tau"], mcfg.get("seed", 0)
    batch_seqs = int(mcfg.get("batch_seqs", 512))
    inner_iters = int(mcfg.get("inner_iters", 8))
    eval_samples = int(mcfg.get("eval_samples", 96))
    tax = build_taxonomy()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)   # created up-front so each pass can checkpoint

    def V(task, allowed=None):
        return CircuitVerifier(model, task, tau=tau, allowed=allowed, base_seed=seed,
                               inner_iters=inner_iters, batch_seqs=batch_seqs)

    reports: dict[str, dict] = {}

    # ---- Pass 1: families -------------------------------------------------------------------
    print("\n=== PASS 1: family ===", flush=True)
    # doability filter: the model must actually do the family task (positive correct-foil margin).
    # full_metric is now the KL signal (always > 0), so the filter uses full_margin instead.
    fam_v = {f: v for f, n in tax.items() if (v := V(n["family"])).full_margin > 1e-6}
    if not fam_v:
        print("[abort] no family has a positive full-model margin", flush=True)
        return 1
    fam_runner, reports["family"] = _amortized(model, list(fam_v.values()), mcfg, label="family",
                                               evaluate=False)   # coarse level: support only, no eval
    fam_support = {f: _support(fam_runner, v, eval_samples) for f, v in fam_v.items()}
    torch.save({"policy_state_dict": fam_runner.actor.state_dict(), "support": fam_support,
                "n_comp": model.n_comp}, out / "family.pt")
    print(f"[ckpt] {out / 'family.pt'} (family policy + supports)", flush=True)

    # ---- Pass 2: subtasks, restricted to the family support, warm-started from family --------
    print("\n=== PASS 2: subtask (in family support) ===", flush=True)
    sub_v: dict[tuple, CircuitVerifier] = {}
    for fam, node in tax.items():
        supp = fam_support.get(fam, 0)
        if not supp:
            continue
        for sub, snode in node["subtasks"].items():
            v = V(snode["subtask"], allowed=supp)
            if v.s0(supp):
                sub_v[(fam, sub)] = v
            else:
                print(f"[skip] {fam}/{sub}: family support not faithful", flush=True)
    sub_runner, sub_support = fam_runner, {}
    if sub_v:
        sub_runner, reports["subtask"] = _amortized(model, list(sub_v.values()), mcfg, label="subtask",
                                                    init_actor_state=fam_runner.actor.state_dict(),
                                                    evaluate=False)   # coarse level: support only
        sub_support = {k: _support(sub_runner, v, eval_samples) for k, v in sub_v.items()}
        torch.save({"policy_state_dict": sub_runner.actor.state_dict(),
                    "support": {f"{f}/{s}": m for (f, s), m in sub_support.items()},
                    "n_comp": model.n_comp}, out / "subtask.pt")
        print(f"[ckpt] {out / 'subtask.pt'} (subtask policy + supports)", flush=True)
    else:
        print("[warn] Pass 2 empty: no subtask faithful in its family support", flush=True)

    # ---- Pass 3: subsubtasks (leaves), restricted to the subtask support, warm-started -------
    print("\n=== PASS 3: subsubtask (in subtask support) ===", flush=True)
    leaf_v: dict[tuple, CircuitVerifier] = {}
    for fam, node in tax.items():
        for sub, snode in node["subtasks"].items():
            supp = sub_support.get((fam, sub), 0)
            if not supp:
                continue
            for leaf in snode["leaves"]:
                v = V(leaf, allowed=supp)
                if v.s0(supp):
                    leaf_v[(fam, sub, leaf.name)] = v
                else:
                    print(f"[skip] {fam}/{sub}/{leaf.name}: subtask support not faithful", flush=True)
    leaf_runner, leaf_circuit = sub_runner, {}
    if leaf_v:
        leaf_runner, reports["subsubtask"] = _amortized(model, list(leaf_v.values()), mcfg,
                                                        label="subsubtask",  # leaf: the deliverable,
                                                        init_actor_state=sub_runner.actor.state_dict())
        leaf_circuit = {k: _support(leaf_runner, v, eval_samples) for k, v in leaf_v.items()}
        torch.save({"policy_state_dict": leaf_runner.actor.state_dict(),
                    "circuits": {f"{f}/{s}/{lf}": m for (f, s, lf), m in leaf_circuit.items()},
                    "n_comp": model.n_comp}, out / "leaf.pt")
        print(f"[ckpt] {out / 'leaf.pt'} (leaf policy + circuits)", flush=True)
    else:
        print("[warn] Pass 3 empty: no leaf faithful in its subtask support", flush=True)

    # ---- hash table + decomposition report --------------------------------------------------
    hashtable: dict = {}
    for (fam, sub, leaf), circ in leaf_circuit.items():
        hashtable.setdefault(fam, {}).setdefault(sub, {})[leaf] = circ
    print(f"\n{'level':10s} {'name':22s} {'#found':>7s} {'faith':>6s} {'size':>6s}")
    for fam in fam_v:  # coarse levels report support size only (no eval was run)
        print(f"{'family':10s} {fam:22s} {'-':>7s} {'-':>6s} "
              f"{bin(fam_support.get(fam, 0)).count('1'):6d}")
    for (fam, sub) in sub_v:
        print(f"{'subtask':10s} {fam + '/' + sub:22s} {'-':>7s} {'-':>6s} "
              f"{bin(sub_support.get((fam, sub), 0)).count('1'):6d}")
    for (fam, sub, leaf), circ in leaf_circuit.items():
        r = reports["subsubtask"][leaf]
        print(f"{'leaf':10s} {fam + '/' + sub + '/' + leaf:22s} {r['n_found']:7d} "
              f"{r['mean_faithfulness']:6.2f} {r['median_size']:6.0f}  ({bin(circ).count('1')} comp)")

    (out / "hashtable.json").write_text(json.dumps(
        {"hashtable_sizes": {f: {s: {lf: bin(c).count("1") for lf, c in lv.items()}
                                 for s, lv in sv.items()} for f, sv in hashtable.items()},
         # the actual circuit masks, keyed "family/subtask/leaf", so the router can load and
         # activate each leaf's sparse circuit for the sparse-vs-dense routed evaluation.
         "leaf_masks": {f"{fam}/{sub}/{leaf}": circ
                        for (fam, sub, leaf), circ in leaf_circuit.items()},
         "family_support_size": {f: bin(s).count("1") for f, s in fam_support.items()},
         "reports": reports}, indent=2))
    torch.save({"policy_state_dict": leaf_runner.actor.state_dict(), "n_comp": model.n_comp},
               out / "leaf_policy.pt")
    print(f"\n[done] {out/'hashtable.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
