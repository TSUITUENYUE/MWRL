"""Hierarchical MWRL for circuits: a coarse family circuit refined to fine subtask circuits.

Two sequential amortized MWRL passes (paper Definition 1 at two granularities -- no nested MDP,
which would be a loop-complexity blowup):

  1. Family pass: MWRL over the merged family tasks (a family's prompts are the union of its
     subtasks') -> the family antichain. The family SUPPORT is the union of the recovered family
     circuits: the components that family relies on (coarse).
  2. Subtask pass: MWRL over the subtasks with D(c) restricted to the parent family's support
     (via CircuitVerifier.allowed) -> a finer circuit living inside the family circuit.

Result: family circuit (coarse) contains subtask circuit (fine), a two-level decomposition.

    python -m mwrl_circuits.hierarchical --config configs/qwen3_1p7b.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from mwrl.runner import Runner

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.env import CircuitEnv
from mwrl_circuits.evaluate import evaluate_antichain
from mwrl_circuits.run import METHODS, build_config
from mwrl_circuits.tasks import build_hierarchy
from mwrl_circuits.verifier import CircuitVerifier


def _amortized(model: AblatedModel, verifiers: list[CircuitVerifier], mcfg: dict, *, label: str,
               init_actor_state: dict | None = None, evaluate: bool = True):
    """One amortized MWRL pass over the given verifiers; returns (runner, {name: eval report}).

    ``init_actor_state`` warm-starts the policy from a prior pass/run instead of cold-starting --
    reuse across granularity IS the amortization claim, and (n_comp fixed) the actor is shape-
    compatible across passes, so the family policy transfers directly to subtask refinement.

    ``evaluate=False`` skips the per-verifier antichain scoring. The coarse passes (family, subtask)
    only need the runner for _support (the restriction that feeds the next pass); their eval report
    is used nowhere downstream, so scoring them is pure -- and unbatched -- overhead. Only the leaf
    pass, whose report is the deliverable, evaluates.
    """
    config, horizon, seed, eval_samples = build_config(mcfg)
    advantage, optimizer = METHODS["mwrl"]
    env = CircuitEnv(verifiers, n_comp=model.n_comp, horizon=horizon or model.n_comp)
    runner = Runner(env, config, advantage=advantage, optimizer=optimizer, device=model.device, seed=seed)
    if init_actor_state is not None:
        runner.actor.load_state_dict(init_actor_state)
        print(f"[{label}] warm-started from a prior policy (no cold start)", flush=True)
    t0 = time.time()

    def progress(u: int, sr: float) -> None:
        if u % max(1, config.updates // 10) == 0 or u == config.updates - 1:
            print(f"[{label} update {u + 1}/{config.updates}] success={sr:.2f} {time.time()-t0:.0f}s",
                  flush=True)

    runner.train(on_update=progress)
    report = {}
    if evaluate:
        for v in verifiers:
            m = evaluate_antichain(runner, v, n_samples=eval_samples)
            m["circuits"] = None  # drop the raw masks from the printed report; keep the union below
            report[v.name] = m
    return runner, report


def _support(runner: Runner, verifier: CircuitVerifier, n_samples: int) -> int:
    """Union of the recovered minimal circuits for this behavior: its component support. Uses the
    batched group rollout so the n_samples terminal searches ride one batched inner-optimizer pass."""
    support = 0
    for roll in runner._rollout_group(verifier, n_samples):
        if roll.success:
            support |= roll.canon
    return support


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="runs/circuits/hierarchical")
    ap.add_argument("--warm-start", default=None,
                    help="checkpoint.pt (with policy_state_dict) to warm-start Pass 1 from a prior run")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg = dict(cfg.get("mwrl", {}))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    tau, seed = cfg["tau"], mcfg.get("seed", 0)
    batch_seqs = int(mcfg.get("batch_seqs", 1024))
    inner_iters = int(mcfg.get("inner_iters", 8))
    eval_samples = int(mcfg.get("eval_samples", 96))
    hierarchy = build_hierarchy(cfg["families"])

    # ---- Pass 1: family circuits ------------------------------------------------------------
    print("\n=== PASS 1: family-level MWRL ===", flush=True)
    fam_verifiers, fam_of = [], {}
    for fam, node in hierarchy.items():
        v = CircuitVerifier(model, node["family"], tau=tau, base_seed=seed,
                            inner_iters=inner_iters, batch_seqs=batch_seqs)
        if v.full_metric <= 1e-6:
            print(f"[skip] family {fam}: nonpositive margin", flush=True)
            continue
        fam_verifiers.append(v)
        fam_of[fam] = v
    warm = None
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location=model.device,
                          weights_only=False)["policy_state_dict"]
        print(f"[warm-start] Pass 1 from {args.warm_start}", flush=True)
    fam_runner, fam_report = _amortized(model, fam_verifiers, mcfg, label="family",
                                        init_actor_state=warm)
    supports = {fam: _support(fam_runner, v, eval_samples) for fam, v in fam_of.items()}

    # ---- Pass 2: subtask circuits restricted to the parent family support --------------------
    print("\n=== PASS 2: subtask-level MWRL, restricted to the family support ===", flush=True)
    sub_verifiers, sub_family = [], {}
    for fam, node in hierarchy.items():
        support = supports.get(fam, 0)
        if not support:
            continue
        for sub in node["subtasks"]:
            v = CircuitVerifier(model, sub, tau=tau, allowed=support, base_seed=seed,
                                inner_iters=inner_iters, batch_seqs=batch_seqs)
            if not v.s0(v.allowed):   # the family support must still contain a faithful sub-circuit
                print(f"[skip] {sub.name}: family {fam} support not faithful for it", flush=True)
                continue
            sub_verifiers.append(v)
            sub_family[sub.name] = fam
    # Pass 2 warm-starts from the FAMILY policy -- the family circuit's selection transfers to
    # subtask refinement, which is the hierarchical amortization claim (not a cold re-train).
    sub_runner, sub_report = _amortized(model, sub_verifiers, mcfg, label="subtask",
                                        init_actor_state=fam_runner.actor.state_dict())

    # ---- report the two-level decomposition -------------------------------------------------
    print(f"\n{'level':8s} {'name':16s} {'parent':14s} {'#found':>7s} {'size':>6s} {'support':>8s}")
    for fam, v in fam_of.items():
        r = fam_report[fam]
        print(f"{'family':8s} {fam:16s} {'-':>14s} {r['n_found']:7d} {r['median_size']:6.0f} "
              f"{bin(supports[fam]).count('1'):8d}")
    for name, r in sub_report.items():
        print(f"{'subtask':8s} {name:16s} {sub_family[name]:14s} {r['n_found']:7d} "
              f"{r['median_size']:6.0f} {'-':>8s}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(
        {"family": fam_report, "subtask": sub_report,
         "family_support_size": {f: bin(s).count("1") for f, s in supports.items()},
         "family_support_mask": dict(supports.items())}, indent=2))  # masks, not just sizes -> reusable
    # save the trained policies so a redo / next benchmark warm-starts instead of cold-training.
    torch.save({"policy_state_dict": fam_runner.actor.state_dict(), "level": "family",
                "n_comp": model.n_comp}, out / "family_policy.pt")
    torch.save({"policy_state_dict": sub_runner.actor.state_dict(), "level": "subtask",
                "n_comp": model.n_comp}, out / "subtask_policy.pt")
    print(f"\n[done] {out/'report.json'}  (+ family_policy.pt, subtask_policy.pt)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
