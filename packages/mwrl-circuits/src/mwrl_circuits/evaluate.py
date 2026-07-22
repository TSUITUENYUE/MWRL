"""Open-world antichain scoring for circuits: no ground-truth antichain, so score the runner's
canonicalized witnesses by born-minimality (soundness), the number of distinct minimal circuits
recovered (#found, uncapped), and faithfulness. A witness is born-minimal if no single kept
component is removable while it stays a witness."""

from __future__ import annotations

import torch
from mwrl.credit import popcount
from mwrl.runner import Runner

from mwrl_circuits.verifier import CircuitVerifier


def is_locally_minimal(verifier: CircuitVerifier, mask: int) -> bool:
    # born-minimal w.r.t. the base predicate s0: no single kept component is removable while the
    # exact masked sub-network stays faithful. One batched forward over all single removals.
    comps = [i for i in range(verifier.n_comp) if (mask >> i) & 1]
    if not comps:
        return False
    return not any(verifier.s0_batch([mask & ~(1 << i) for i in comps]))


@torch.no_grad()
def evaluate_antichain(runner: Runner, verifier: CircuitVerifier, *, n_samples: int = 96) -> dict:
    # batched group rollout: the n_samples terminal searches share one batched inner-optimizer pass,
    # exactly like _support -- NOT n_samples sequential runner.rollout calls (that unbatched loop was
    # the CPU-bound eval bottleneck).
    rolls = runner._rollout_group(verifier, n_samples)
    witnesses = [r.canon for r in rolls if r.success]   # canonical minimal witness Phi, not raw S_T
    sizes = [popcount(w) for w in witnesses]
    distinct = list(dict.fromkeys(witnesses))     # the recovered antichain, uncapped
    minimal = [m for m in distinct if is_locally_minimal(verifier, m)]
    faith = [verifier.faithfulness(m) for m in minimal]
    return {
        "soundness": len(minimal) / max(len(distinct), 1),
        "n_found": len(minimal),
        "n_distinct": len(distinct),
        "witness_rate": len(witnesses) / max(n_samples, 1),
        "median_size": float(sorted(sizes)[len(sizes) // 2]) if sizes else 0.0,
        "mean_faithfulness": float(sum(faith) / len(faith)) if faith else 0.0,
        "circuits": minimal,
    }
