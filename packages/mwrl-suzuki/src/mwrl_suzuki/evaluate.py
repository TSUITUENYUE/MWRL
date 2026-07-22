"""Closed-world antichain scoring against the exact ground-truth antichain ``M(c)``.

Because the dataset defines ``M(c)`` exactly, completeness is measurable here (unlike the
open-world circuits setting). Soundness is the born-minimal rate of the raw terminal proposals
(no post-hoc pruning); completeness and #found count the distinct raw terminals that are
exactly minimal witnesses, the fraction of ``M(c)`` covered. This is the strict raw-hit
convention of the MaxSAT coverage rows: a terminal that merely contains a minimal witness
earns nothing."""

from __future__ import annotations

import numpy as np
import torch
from mwrl.credit import popcount
from mwrl.runner import Runner

from mwrl_suzuki.verifier import ChemVerifier


@torch.no_grad()
def evaluate_antichain(
    runner: Runner,
    verifier: ChemVerifier,
    *,
    n_samples: int = 128,
    record_proposals: bool = False,
) -> dict:
    antichain = set(verifier.ground_truth_antichain())
    n_succ = born = 0
    recovered: set[int] = set()
    sizes: list[int] = []
    proposals: list[list[int]] = []
    for _ in range(n_samples):
        roll = runner.rollout(verifier)
        if record_proposals:
            # rollout order is the policy's implicit ranking; keeping every
            # attempt (successful or not) lets any proposal-budget convention
            # be scored post hoc without rerunning the policy
            proposals.append([
                int(roll.witness),
                int(roll.canon) if roll.success else -1,
                int(roll.success),
            ])
        if not roll.success:
            continue
        n_succ += 1
        sizes.append(popcount(roll.witness))
        if roll.witness in antichain:          # raw terminal already minimal (born-minimal)
            born += 1
        if roll.canon in antichain:            # minimal witness recovered via canonicalization
            recovered.add(roll.canon)
    faith = [verifier.faithfulness(w) for w in recovered]
    if record_proposals:
        return {
            "proposals": proposals,
            "antichain": sorted(antichain),
            **_metrics(antichain, n_succ, born, recovered, sizes, faith,
                       n_samples, verifier),
        }
    return _metrics(antichain, n_succ, born, recovered, sizes, faith,
                    n_samples, verifier)


def _metrics(antichain, n_succ, born, recovered, sizes, faith, n_samples, verifier):
    return {
        "antichain_size": len(antichain),
        "soundness": born / max(n_succ, 1),
        "completeness": len(recovered) / max(len(antichain), 1),
        "n_found": len(recovered),
        "success_rate": n_succ / max(n_samples, 1),
        "median_size": float(sorted(sizes)[len(sizes) // 2]) if sizes else 0.0,
        "mean_faithfulness": float(np.mean(faith)) if faith else 0.0,
        "forward_calls": verifier.forward_calls,
    }
