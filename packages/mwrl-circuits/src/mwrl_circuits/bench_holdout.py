"""Held-out circuit evaluation: do the discovered circuits generalize to new prompts?

Circuits were discovered on 12 probe questions per subject. This script evaluates the
FROZEN masks on a disjoint holdout (disjoint by dataset row id, see bench.mmlu_holdout_task)
and reports, per subject:

  sufficiency   s_test bit at the same tau, with absolute KL and KL/KL_empty so the
                renormalized threshold cannot conceal a weak signal
  accuracy      real 4-choice accuracy, greedy exact match, and the pairwise margin,
                for dense / MWRL / EAP / magnitude / random at matched size
  minimality    only for holdout-SUFFICIENT circuits: the fraction of components whose
                deletion breaks holdout sufficiency (1.0 = every component critical);
                insufficiency and non-minimality are never conflated
  family        the fraction of ALL recovered circuits (not just the representative)
                that remain holdout-sufficient
  confusion     the 29x29 task-by-circuit 4-choice accuracy matrix (representative masks)

Frozen-mask rule: the EAP mask uses DISCOVERY attributions only; magnitude uses weights;
random masks use fixed seeds; MWRL masks come from the discovery report. Nothing reads the
holdout before masks are fixed. Summary means carry paired-bootstrap 95% CIs over subjects.

    python -m mwrl_circuits.bench_holdout --config configs/qwen3_1p7b_bench.yaml \
        --circuits runs/.../report.json --output runs/.../bench_holdout.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import yaml

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.baselines_masks import acdc_mask, hardconcrete_mask, wanda_scores
from mwrl_circuits.bench import MMLU_SUBJECTS, mmlu_holdout_task, mmlu_task
from mwrl_circuits.bench_eval import _magnitude_scores, _random_mask, _topk_mask
from mwrl_circuits.router import _mask_from_names
from mwrl_circuits.verifier import CircuitVerifier

_CAVEAT = ("subjects were selected by dense accuracy on the same MMLU test split, so the "
           "holdout is unseen by circuit discovery but conditional on dense-capable subjects")


@torch.no_grad()
def _accs(v: CircuitVerifier, keep_mask: int | None, letter_ids: torch.Tensor) -> dict:
    """4-choice accuracy, full-vocab greedy exact match, and pairwise margin accuracy."""
    keep = None if keep_mask is None else v._keep(keep_mask)
    logits = v.model.logits(v.enc, keep, v.cf)                     # [P, vocab]
    correct_col = (v.correct.unsqueeze(1) == letter_ids.unsqueeze(0)).float().argmax(1)
    acc4 = (logits[:, letter_ids].argmax(-1) == correct_col).float().mean()
    greedy = (logits.argmax(-1) == v.correct).float().mean()
    pair = (logits[v.rows, v.correct] > logits[v.rows, v.foil]).float().mean()
    return {"acc4": float(acc4), "greedy": float(greedy), "pair": float(pair)}


@torch.no_grad()
def _kls(v: CircuitVerifier, masks: list[int]) -> list[float]:
    kl = v.model.kl_batch(v.enc, v._keeps(masks), v.clean_logits, v.cf, chunk=v._batch_masks)
    return [float(x) for x in kl]


def _boot(values: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    """Paired bootstrap over subjects: 95% CI of each column's mean and of key differences."""
    rng = np.random.default_rng(seed)
    names = list(values)
    n = len(next(iter(values.values())))
    idx = rng.integers(0, n, size=(n_boot, n))
    out: dict[str, dict] = {}
    boots = {m: values[m][idx].mean(axis=1) for m in names}
    for m in names:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        out[m] = {"mean": float(values[m].mean()), "ci95": [float(lo), float(hi)]}
    for m in names:
        if m in ("mwrl", "dense"):
            continue
        d = boots["mwrl"] - boots[m]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[f"mwrl_minus_{m}"] = {"mean": float((values["mwrl"] - values[m]).mean()),
                                  "ci95": [float(lo), float(hi)]}
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--circuits", required=True, help="report.json from the discovery run")
    ap.add_argument("--output", required=True)
    ap.add_argument("--holdout-k", type=int, default=24)
    ap.add_argument("--holdout-seed", type=int, default=1)
    ap.add_argument("--random-masks", type=int, default=5)
    ap.add_argument("--boots", type=int, default=10000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg = cfg.get("mwrl", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
                         attn_implementation=cfg.get("attn_implementation", "eager"))
    report = json.load(open(args.circuits))
    magnitude = _magnitude_scores(model)
    letter_ids = torch.tensor(
        [model.tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in "ABCD"],
        device=model.device)
    batch_seqs = int(mcfg.get("batch_seqs", 1024))

    rows: dict[str, dict] = {}
    hold_verifiers: dict[str, CircuitVerifier] = {}
    rep_masks: dict[str, int] = {}
    for i, subject in enumerate(MMLU_SUBJECTS):
        name = f"mmlu_{subject}"
        entry = report.get(name)
        if not entry or not entry.get("circuits") or not min(entry["circuits"], key=len):
            print(f"[skip] {name}: no recovered circuit", flush=True)
            continue
        # ---- freeze every mask from discovery-side information only -------------------
        disc_task = mmlu_task(subject)
        v_disc = CircuitVerifier(model, disc_task, tau=cfg["tau"], batch_seqs=batch_seqs)
        family = sorted({_mask_from_names(c, model) for c in entry["circuits"]})
        rep = _mask_from_names(min(entry["circuits"], key=len), model)
        k = bin(rep).count("1")
        masks = {"mwrl": rep,
                 "eap": _topk_mask(v_disc.attrs.abs().cpu(), k),
                 "magnitude": _topk_mask(magnitude, k),
                 "wanda": _topk_mask(wanda_scores(model, v_disc), k),
                 "hardconcrete": hardconcrete_mask(model, v_disc, k),
                 "acdc": acdc_mask(v_disc)}
        randoms = [_random_mask(model.n_comp, k, 1000 * i + j) for j in range(args.random_masks)]
        # ---- only now touch the holdout ----------------------------------------------
        task, hold_ids, disc_ids = mmlu_holdout_task(subject, k=args.holdout_k,
                                                     seed=args.holdout_seed)
        v = CircuitVerifier(model, task, tau=cfg["tau"], batch_seqs=batch_seqs)
        hold_verifiers[name], rep_masks[name] = v, rep
        ordered = list(masks.values()) + randoms + family
        kl = _kls(v, ordered)
        thresh = (1.0 - cfg["tau"]) * v.kl_empty
        row = {"size": k, "kl_empty": v.kl_empty, "discovery_ids": disc_ids,
               "holdout_ids": hold_ids, "n_holdout": len(task.prompts),
               "dense": _accs(v, None, letter_ids)}
        for pos, m in enumerate(masks):
            row[m] = _accs(v, masks[m], letter_ids)
            row[m].update({"kl": kl[pos], "kl_norm": kl[pos] / v.kl_empty,
                           "sufficient": kl[pos] <= thresh,
                           "size": bin(masks[m]).count("1")})
        rnd = [_accs(v, r, letter_ids) for r in randoms]
        rnd_kl = kl[len(masks):len(masks) + len(randoms)]
        row["random"] = {key: float(np.mean([x[key] for x in rnd]))
                         for key in ("acc4", "greedy", "pair")}
        row["random"].update({"kl": float(np.mean(rnd_kl)),
                              "kl_norm": float(np.mean(rnd_kl)) / v.kl_empty,
                              "sufficient": bool(np.mean([x <= thresh for x in rnd_kl]) > 0.5)})
        fam_kl = kl[len(masks) + len(randoms):]
        row["family"] = {"n": len(family),
                         "frac_sufficient": float(np.mean([x <= thresh for x in fam_kl]))}
        # minimality only where sufficiency holds; never conflate the two failures
        if row["mwrl"]["sufficient"] and k > 0:
            deletions = [rep & ~(1 << b) for b in range(model.n_comp) if (rep >> b) & 1]
            suff = v.s0_batch(deletions)
            row["mwrl"]["deletion_critical"] = float(np.mean([not s for s in suff]))
        else:
            row["mwrl"]["deletion_critical"] = None
        rows[name] = row
        print(f"{name:34s} k={k:3d} dense4={row['dense']['acc4']:.2f} "
              f"mwrl4={row['mwrl']['acc4']:.2f} eap4={row['eap']['acc4']:.2f} "
              f"acdc4={row['acdc']['acc4']:.2f} suff={int(row['mwrl']['sufficient'])} "
              f"famS={row['family']['frac_sufficient']:.2f}", flush=True)
        json.dump({"meta": {"caveat": _CAVEAT, "partial": True}, "rows": rows},
                  open(args.output + ".partial", "w"), indent=1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    names = list(hold_verifiers)
    if not names:
        json.dump({"meta": {"caveat": _CAVEAT}, "rows": {}, "summary": {}},
                  open(args.output, "w"), indent=1)
        print(f"[done] nothing to evaluate; empty {args.output}", flush=True)
        return 0
    print("\n[confusion] holdout 4-choice accuracy, task_i on circuit_j", flush=True)
    confusion = [[_accs(hold_verifiers[a], rep_masks[b], letter_ids)["acc4"] for b in names]
                 for a in names]
    diag = [confusion[i][i] for i in range(len(names))]
    off = [confusion[i][j] for i in range(len(names)) for j in range(len(names)) if i != j]

    cols = {m: np.array([rows[n][m]["acc4"] for n in names])
            for m in ("dense", "mwrl", "eap", "magnitude", "random", "wanda",
                      "hardconcrete", "acdc")}
    summary = {"acc4": _boot(cols, args.boots, seed=0),
               "sufficient_frac": float(np.mean([rows[n]["mwrl"]["sufficient"] for n in names])),
               "family_sufficient_frac": float(np.mean(
                   [rows[n]["family"]["frac_sufficient"] for n in names])),
               "deletion_critical_mean": float(np.mean(
                   [rows[n]["mwrl"]["deletion_critical"] for n in names
                    if rows[n]["mwrl"]["deletion_critical"] is not None] or [float("nan")])),
               "confusion_diag": float(np.mean(diag)), "confusion_off": float(np.mean(off))}
    print("\n[summary]", flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    json.dump({"meta": {"tau": cfg["tau"], "holdout_k": args.holdout_k,
                        "holdout_seed": args.holdout_seed,
                        "random_masks": args.random_masks, "caveat": _CAVEAT},
               "rows": rows, "confusion": {"names": names, "matrix": confusion},
               "summary": summary}, open(args.output, "w"), indent=1)
    print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
