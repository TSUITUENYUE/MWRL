"""Deliverable metrics: judge a policy by the antichain it returns, not by its success rate.

For one held-out instance with B sampled answers:
  success       fraction of answers that determine x with the right value
  soundness     fraction of successful answers that are minimal (1 - superset rate)
  found         number of distinct minimal sets among the B answers
  recall@k      expected fraction of the antichain found by k of the B answers, drawn
                without replacement (unbiased, like pass@k)
  pass@k        probability that k answers contain at least one success
"""

from __future__ import annotations

from math import comb

import numpy as np

from mwrl_rlvr.ds import Verdict


def _hit_at_k(total: int, hits: int, k: int) -> float:
    """P(at least one of `hits` marked items among k drawn without replacement from `total`)."""
    if hits <= 0:
        return 0.0
    if total - hits < k:
        return 1.0
    return 1.0 - comb(total - hits, k) / comb(total, k)


def instance_metrics(verdicts: list[Verdict], antichain: list[int], ks: list[int]) -> dict:
    b = len(verdicts)
    succ = [v for v in verdicts if v.success]
    per_min = {m: sum(1 for v in succ if v.mask == m) for m in antichain}
    out = {
        "B": b,
        "antichain": len(antichain),
        "parsed": sum(v.parsed for v in verdicts) / b,
        "success": len(succ) / b,
        "n_success": len(succ),
        "n_minimal": sum(v.minimal for v in succ),
        "soundness": (sum(v.minimal for v in succ) / len(succ)) if succ else None,
        "found": sum(1 for c in per_min.values() if c > 0),
        "mean_size_success": float(np.mean([v.size for v in succ])) if succ else None,
        "distinct_success_sets": len({v.mask for v in succ}),
    }
    for k in ks:
        if k > b:
            continue
        out[f"pass@{k}"] = _hit_at_k(b, len(succ), k)
        out[f"recall@{k}"] = float(np.mean([_hit_at_k(b, c, k) for c in per_min.values()]))
    return out


def aggregate(rows: list[dict]) -> dict:
    """Mean over instances (soundness and mean size over instances with a success), plus the
    pooled soundness: all minimal successes over all successes."""
    keys = [k for k in rows[0] if k not in ("B", "n_success", "n_minimal")]
    out: dict = {"instances": len(rows), "B": rows[0]["B"]}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    n_succ = sum(r["n_success"] for r in rows)
    out["soundness_pooled"] = sum(r["n_minimal"] for r in rows) / n_succ if n_succ else None
    return out
