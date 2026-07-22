"""Measure the non-monotonicity of the circuit verifier (raw s0 and the realized closure).

The theory assumes a monotone witness test. The raw predicate s0 (keep-only-S reproduces the
next-token distribution within (1-tau) of KL_empty) is non-monotone, and the pipeline bridges
the gap with the finite existential closure s_exists (inner EAP-dropout search). This script
quantifies the bridge on the actual benchmark verifiers:

  1. witnesses      Phi-certificates from the pipeline's own search (s_exists_batch), from the
                    full mask and random dense starts -- the certified faithful circuits.
  2. raw violations for each witness w with s0(w)=1, random supersets T = w | (k extras) probe
                    monotonicity: every s0(T)=0 is a violation. Rate by k plus depth
                    (faithfulness of the violator against tau: borderline or deep).
  3. closure repair on violating supersets the ground truth is s_exists(T)=1 (w is inside T),
                    so a failed inner search is a violation of the REALIZED witness test.
                    residual rate = raw rate x (1 - repair rate).
  4. irredundancy   for each certificate, all single deletions are tested; a certificate with a
                    removable element is a false-minimal certificate.

    python -m mwrl_circuits.monotonicity --config configs/qwen3_1p7b_bench.yaml \
        --output runs/circuits/mono_bench10.json
    python -m mwrl_circuits.monotonicity --config configs/qwen3_8b.yaml --taxonomy \
        --levels family,leaf --output runs/circuits/mono_hier8b.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.tasks import build_hierarchy, build_taxonomy
from mwrl_circuits.verifier import CircuitVerifier

def _bits(mask: int) -> list[int]:
    return [i for i in range(mask.bit_length()) if (mask >> i) & 1]


def _kls(v: CircuitVerifier, masks: list[int], slice_: int = 256) -> list[float]:
    """Raw KL(P_clean || P_mask) for every mask, batched exactly like s0_batch."""
    out: list[float] = []
    for i in range(0, len(masks), slice_):
        keeps = v._keeps(masks[i:i + slice_])
        kl = v.model.kl_batch(v.enc, keeps, v.clean_logits, v.cf, chunk=v._batch_masks)
        out.extend(kl.tolist())
    return out


def _witnesses(v: CircuitVerifier, rng: np.random.Generator, want: int) -> list[int]:
    """Phi-certificates from the pipeline's own closure search, from the full mask plus random
    dense starts (the search is deterministic per start mask, so diversity comes from starts)."""
    allowed_bits = _bits(v.allowed)
    starts = [v.allowed]
    for dens in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):
        if len(starts) >= max(2 * want, want + 3):
            break
        keep = rng.random(len(allowed_bits)) < dens
        m = 0
        for b, kp in zip(allowed_bits, keep, strict=True):
            if kp:
                m |= 1 << b
        starts.append(m)
    found = v.s_exists_batch(list(dict.fromkeys(starts)))
    canons = list(dict.fromkeys(c for ok, c in found if ok and c))
    return canons[:want]


def measure_task(v: CircuitVerifier, task_idx: int, args) -> dict:
    rng = np.random.default_rng([args.seed, task_idx])
    thresh = (1.0 - v.tau) * v.kl_empty
    t0 = time.time()
    witnesses = _witnesses(v, rng, args.witnesses)
    if not witnesses:
        return {"skipped": "no witness found", "kl_empty": v.kl_empty}

    # certificate irredundancy: no single deletion of a certificate may still pass s0.
    irred, removable = [], []
    for w in witnesses:
        singles = [w & ~(1 << b) for b in _bits(w)]
        passes = v.s0_batch(singles)
        n_rm = sum(passes)
        removable.append(n_rm)
        irred.append(n_rm == 0)

    # monotonicity probes: supersets of certified witnesses.
    ks = [int(k) for k in args.ks.split(",")]
    probes: list[tuple[int, int, int]] = []              # (witness_idx, k, T)
    for wi, w in enumerate(witnesses):
        comp = [b for b in _bits(v.allowed) if not (w >> b) & 1]
        for k in ks:
            if k > len(comp):
                continue
            for _ in range(args.probes_per_k):
                extra = rng.choice(len(comp), size=k, replace=False)
                t = w
                for e in extra:
                    t |= 1 << comp[int(e)]
                probes.append((wi, k, t))
    uniq = list(dict.fromkeys(t for _, _, t in probes))
    kl_of = dict(zip(uniq, _kls(v, uniq), strict=True))

    by_k: dict[int, dict] = {}
    viol_anchor: dict[int, int] = {}                      # violating T -> its anchor witness w
    for wi, k, t in probes:
        rec = by_k.setdefault(k, {"n": 0, "viol": 0, "faith_viol": []})
        rec["n"] += 1
        if kl_of[t] > thresh:                             # s0(T) = 0: monotonicity violation
            rec["viol"] += 1
            rec["faith_viol"].append(1.0 - kl_of[t] / v.kl_empty)
            viol_anchor.setdefault(t, witnesses[wi])
    for rec in by_k.values():
        fv = rec.pop("faith_viol")
        rec["viol_rate"] = rec["viol"] / rec["n"]
        rec["faith_viol_mean"] = float(np.mean(fv)) if fv else None
        rec["faith_viol_min"] = float(np.min(fv)) if fv else None

    # closure repair: the realized witness test on violating supersets (ground truth = 1).
    sample = list(viol_anchor)[:args.max_repair]
    repaired = sum(ok for ok, _ in v.s_exists_batch(sample)) if sample else 0

    # paired repair sweep over closure budgets: the SAME violating supersets re-searched at each
    # inner_iters level (per-mask RNG is iters-independent, so levels are exactly paired). This is
    # count's monotonicity lever: the certificate contract leans on the closure, so repair should
    # rise with iters; coverage's credit reads raw certified terminals and never queries Phi.
    sweep: dict[str, dict] = {}
    if args.repair_iters and sample:
        saved = v.inner_iters
        for lv in (int(x) for x in args.repair_iters.split(",")):
            v.inner_iters = lv
            v._exists_cache = {}                          # cache is keyed by mask only: must clear
            res = v.s_exists_batch(sample)
            ok_n = sum(ok for ok, _ in res)
            eq = sum(1 for (ok, c), t in zip(res, sample, strict=True)
                     if ok and c == viol_anchor[t])
            sizes = [bin(c).count("1") for ok, c in res if ok]
            sweep[str(lv)] = {"n": len(sample), "repair": ok_n,
                              "repair_rate": ok_n / len(sample),
                              "phi_eq_anchor": eq,
                              "phi_size_median": float(np.median(sizes)) if sizes else None}
        v.inner_iters = saved
        v._exists_cache = {}

    n_probe = sum(r["n"] for r in by_k.values())
    n_viol = sum(r["viol"] for r in by_k.values())
    raw_rate = n_viol / max(n_probe, 1)
    repair_rate = repaired / len(sample) if sample else None
    return {
        "kl_empty": v.kl_empty, "n_witnesses": len(witnesses),
        "witness_sizes": [bin(w).count("1") for w in witnesses],
        "irredundant": sum(irred), "removable_per_witness": removable,
        "probes": n_probe, "violations": n_viol, "raw_violation_rate": raw_rate,
        "by_k": {str(k): by_k[k] for k in sorted(by_k)},
        "repair_tested": len(sample), "repair_ok": repaired, "repair_rate": repair_rate,
        "residual_rate": raw_rate * (1.0 - repair_rate) if repair_rate is not None else None,
        "repair_sweep": sweep,
        "forward_calls": v.forward_calls, "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--taxonomy", action="store_true",
                    help="use build_taxonomy (the hierarchical tree) instead of cfg families")
    ap.add_argument("--levels", default="family,leaf",
                    help="taxonomy levels to measure (family,subtask,leaf)")
    ap.add_argument("--witnesses", type=int, default=4, help="certificates probed per behavior")
    ap.add_argument("--probes-per-k", type=int, default=6)
    ap.add_argument("--ks", default="1,2,4,8,16,32,64", help="extra components per superset probe")
    ap.add_argument("--max-repair", type=int, default=12,
                    help="violating supersets re-tested with the full closure search per behavior")
    ap.add_argument("--repair-iters", default="",
                    help="comma-separated inner_iters levels for the paired repair sweep on the "
                         "same violators (count's monotonicity lever), e.g. 1,2,4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="runs/circuits/monotonicity.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = dict(cfg.get("mwrl", {}))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    tau, seed = cfg["tau"], int(mcfg.get("seed", 0))
    batch_seqs = int(mcfg.get("batch_seqs", 1024))
    inner_iters = int(mcfg.get("inner_iters", 8))

    if args.taxonomy:
        levels = [s.strip() for s in args.levels.split(",")]
        tasks: list[tuple[str, object]] = []
        for node in build_taxonomy().values():
            if "family" in levels:
                tasks.append(("family", node["family"]))
            for snode in node["subtasks"].values():
                if "subtask" in levels:
                    tasks.append(("subtask", snode["subtask"]))
                if "leaf" in levels:
                    tasks.extend(("leaf", lf) for lf in snode["leaves"])
    else:
        tasks = [("subtask", t) for node in build_hierarchy(cfg["families"]).values()
                 for t in node["subtasks"]]
    print(f"[monotonicity] {len(tasks)} behaviors, tau={tau}, witnesses={args.witnesses}, "
          f"ks={args.ks}, probes/k={args.probes_per_k}", flush=True)

    report: dict = {"tasks": {}}
    for ti, (level, task) in enumerate(tasks):
        v = CircuitVerifier(model, task, tau=tau, base_seed=seed,
                            inner_iters=inner_iters, batch_seqs=batch_seqs)
        if v.kl_empty <= 1e-9:
            print(f"[skip] {task.name}: no interchange signal", flush=True)
            report["tasks"][task.name] = {"level": level, "skipped": "kl_empty ~ 0"}
            continue
        m = measure_task(v, ti, args)
        m["level"] = level
        report["tasks"][task.name] = m
        if "skipped" in m:
            print(f"[skip] {task.name}: {m['skipped']}", flush=True)
            continue
        print(f"[{ti + 1}/{len(tasks)}] {level:8s} {task.name:26s} "
              f"wit={m['n_witnesses']} irred={m['irredundant']}/{m['n_witnesses']} "
              f"raw_viol={m['raw_violation_rate']:.3f} "
              f"repair={m['repair_ok']}/{m['repair_tested']} "
              f"residual={m['residual_rate'] if m['residual_rate'] is not None else float('nan'):.3f} "
              f"({m['seconds']}s)", flush=True)
        if m.get("repair_sweep"):
            curve = " | ".join(f"L{lv}:{r['repair']}/{r['n']} eq{r['phi_eq_anchor']}"
                               for lv, r in m["repair_sweep"].items())
            print(f"          sweep {curve}", flush=True)

    done = [m for m in report["tasks"].values() if "skipped" not in m]
    if done:
        probes = sum(m["probes"] for m in done)
        viols = sum(m["violations"] for m in done)
        rep_n = sum(m["repair_tested"] for m in done)
        rep_ok = sum(m["repair_ok"] for m in done)
        wit = sum(m["n_witnesses"] for m in done)
        raw = viols / max(probes, 1)
        repair = rep_ok / rep_n if rep_n else None
        report["aggregate"] = {
            "behaviors": len(done), "witnesses": wit,
            "irredundant_rate": sum(m["irredundant"] for m in done) / max(wit, 1),
            "probes": probes, "violations": viols, "raw_violation_rate": raw,
            "repair_tested": rep_n, "repair_rate": repair,
            "residual_rate": raw * (1.0 - repair) if repair is not None else None,
        }
        print(f"\n[aggregate] behaviors={len(done)} witnesses={wit} "
              f"irredundant={report['aggregate']['irredundant_rate']:.3f} "
              f"raw_violation={raw:.4f} repair={repair if repair is not None else float('nan'):.3f} "
              f"residual={report['aggregate']['residual_rate'] if repair is not None else float('nan'):.4f}",
              flush=True)
        levels: dict[str, list[int]] = {}
        for m in done:
            for lv, r in m.get("repair_sweep", {}).items():
                a = levels.setdefault(lv, [0, 0, 0])
                a[0] += r["n"]; a[1] += r["repair"]; a[2] += r["phi_eq_anchor"]
        if levels:
            report["aggregate"]["repair_sweep"] = {
                lv: {"n": n, "repair_rate": rp / n, "phi_eq_anchor_rate": eq / n}
                for lv, (n, rp, eq) in levels.items()}
            print("[sweep] paired closure repair vs inner_iters (same violators):", flush=True)
            for lv, (n, rp, eq) in levels.items():
                print(f"  iters={lv:>3s}  repair={rp / n:.3f}  phi==anchor={eq / n:.3f}  (n={n})",
                      flush=True)
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:
        rev = "unknown"
    report["_meta"] = {"model_name": cfg["model_name"], "tau": tau, "seed": args.seed,
                       "witnesses": args.witnesses, "ks": args.ks,
                       "probes_per_k": args.probes_per_k, "max_repair": args.max_repair,
                       "taxonomy": args.taxonomy, "git": rev, "device": device}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
