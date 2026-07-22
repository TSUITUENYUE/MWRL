"""Exact-ground-truth scoring for MaxSAT: roll out the trained policy on the env and
score its RAW terminal witnesses (no canonicalization) against the enumerated antichain
M. Soundness is the born-minimal rate; completeness and number-found measure antichain
recovery. Works with any trained ``Runner`` (the actor is rolled out through the env)."""

from __future__ import annotations

import numpy as np
import torch
from mwrl.credit import popcount
from mwrl.runner import Runner

from mwrl_maxsat.instance import MaxSatInstance


@torch.no_grad()
def evaluate(runner: Runner, instance: MaxSatInstance, antichain: list[int], *,
             n_samples: int = 300) -> dict:
    ac = set(antichain)
    n_succ, born = 0, 0
    distinct: set[int] = set()
    sizes: list[int] = []
    for _ in range(n_samples):
        roll = runner.rollout(instance)   # sample the trained actor through the env
        if not roll.success:
            continue
        n_succ += 1
        sizes.append(popcount(roll.canon))       # canon = raw witness (coverage) or pruned Phi (count)
        if roll.canon in ac:
            born += 1
            distinct.add(roll.canon)
    return {
        "soundness": born / max(n_succ, 1),                       # born-minimal rate (in M, raw)
        "distinct_born_minimal": len(distinct),
        "antichain_size": len(ac),
        "completeness": len(distinct) / max(len(ac), 1),
        "raw_median_size": float(np.median(sizes)) if sizes else 0.0,
        "success_rate": n_succ / max(n_samples, 1),
    }
