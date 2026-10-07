"""Per-group advantages for every arm of the RLVR comparison.

Each rollout carries one integer code (``encode``): the chosen statement mask plus the
parsed / success / minimal flags. An arm maps the K codes of one prompt's group to K
advantages; the policy loss broadcasts each advantage over the rollout's tokens. Every
rule is the core package's own definition (``mwrl.baselines``, ``mwrl.advantage``), so the
RLVR arms and the paper's arms are the same functions.

Arms
----
grpo           r = success; mwrl.baselines.grpo_group_advantages.
maxrl          r = success; mwrl.baselines.maxrl_group_advantages, (r - mean) / (mean + eps).
grpo_size      r = success * (1 - lambda |S|); GRPO. A separable size penalty.
grpo_minimal   r = 1[S is a minimal sufficient set]; GRPO. Needs a minimality oracle.
distinct       count valuation on the raw sets (no canonicalization): 1[success and no other
               success chose the same S], rloo-centered and divided by the mean raw credit
               (mwrl_group_advantages with witness_valuation="count").
distinct_size  distinct credit times (1 - lambda |S|), same centering and scaling.
mwrl           the paper's credit contract: coverage valuation under the geometric product
               measure nu(up S) = p^|S|, deletion credit with the l2o baseline, divided by
               the group's mean |credit| (mwrl_group_advantages, witness_valuation="coverage").
               The coverage is evaluated exactly on the lattice of the group's union of
               successful sets, which equals the core's inclusion-exclusion (tested) and
               stays fast when a group holds many incomparable successes.
"""

from __future__ import annotations

import numpy as np
import torch
from mwrl.baselines import grpo_group_advantages, maxrl_group_advantages
from mwrl.credit import count_advantages

MASK_BITS = 16
PARSED, SUCCESS, MINIMAL = 1 << MASK_BITS, 1 << (MASK_BITS + 1), 1 << (MASK_BITS + 2)
ANTI_SHIFT = MASK_BITS + 3  # 4 bits: the instance's antichain size, for logging recall
ARMS = ("grpo", "maxrl", "grpo_size", "grpo_minimal", "distinct", "distinct_size", "mwrl")


def encode(mask: int, parsed: bool, success: bool, minimal: bool, antichain_size: int = 0) -> int:
    """Pack one verdict into an integer below 2^23 (exact in float32)."""
    assert 0 <= mask < (1 << MASK_BITS) and 0 <= antichain_size < 16
    flags = (PARSED if parsed else 0) | (SUCCESS if success else 0) | (MINIMAL if minimal else 0)
    return mask | flags | (antichain_size << ANTI_SHIFT)


def decode(codes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(mask, parsed, success, minimal); ``antichain_sizes`` reads the logging field."""
    c = np.asarray(codes, dtype=np.int64)
    return c & (PARSED - 1), (c & PARSED) > 0, (c & SUCCESS) > 0, (c & MINIMAL) > 0


def antichain_sizes(codes) -> np.ndarray:
    return (np.asarray(codes, dtype=np.int64) >> ANTI_SHIFT) & 15


def popcounts(masks) -> np.ndarray:
    return np.array([bin(int(m)).count("1") for m in masks], dtype=np.float64)


def coverage_l2o_exact(masks, success, p: float) -> np.ndarray:
    """A_i - b_i of mwrl.credit.coverage_l2o_advantages under nu_upset_geometric(p).

    Coverage only depends on the coordinates in the union U of the successful sets, so the
    product measure is summed exactly over the 2^|U| subsets of U. A_i is the mass covered by
    i alone; the l2o baseline b_i is 1/(K-1) times the mass covered by exactly one success
    other than i (theory section 7.1)."""
    masks = np.asarray(masks, dtype=np.int64)
    success = np.asarray(success, dtype=bool)
    k = masks.size
    universe = 0
    for m, s in zip(masks, success):
        if s:
            universe |= int(m)
    if universe == 0:
        return np.zeros(k)
    coords = [c for c in range(universe.bit_length()) if (universe >> c) & 1]
    local = np.zeros(k, dtype=np.int64)
    for j, c in enumerate(coords):
        local |= ((masks >> c) & 1) << j
    t = np.arange(1 << len(coords), dtype=np.int64)
    size = popcounts(t)
    w = p**size * (1.0 - p) ** (len(coords) - size)
    cover = ((t[:, None] & local[None, :]) == local[None, :]) & success[None, :]
    count = cover.sum(axis=1)
    a = (w[:, None] * (cover & (count[:, None] == 1))).sum(axis=0)
    if k > 1:
        b = (w[:, None] * ((count[:, None] - cover) == 1)).sum(axis=0) / (k - 1)
    else:
        b = np.zeros(k)
    return a - b


def _count_pipeline(credit: np.ndarray, eps: float) -> np.ndarray:
    """mwrl_group_advantages for witness_valuation="count", witness_baseline="rloo",
    witness_scale="mean", scope "group": rloo centering, divided by the mean raw credit."""
    raw_scale = np.abs(credit).mean()
    centered = credit - credit.mean()
    if credit.size > 1:
        centered = centered * (credit.size / (credit.size - 1))
    return centered / (raw_scale + eps)


def group_advantages(arm: str, codes, *, size_lambda: float, nu_p: float, eps: float = 1e-6) -> np.ndarray:
    """K advantages for one prompt's group of K rollouts."""
    masks, _parsed, success, minimal = decode(codes)
    size = popcounts(masks)

    def grpo(r):
        return grpo_group_advantages(torch.as_tensor(r, dtype=torch.float32), epsilon=eps).double().numpy()

    if arm == "grpo":
        return grpo(success.astype(float))
    if arm == "maxrl":
        r = torch.as_tensor(success.astype(float), dtype=torch.float32)
        return maxrl_group_advantages(r, epsilon=eps).double().numpy()
    if arm == "grpo_size":
        return grpo(success * (1.0 - size_lambda * size))
    if arm == "grpo_minimal":
        return grpo(minimal.astype(float))
    if arm in ("distinct", "distinct_size"):
        credit = np.asarray(count_advantages([int(m) for m in masks], [bool(s) for s in success]), dtype=np.float64)
        if arm == "distinct_size":
            credit = credit * (1.0 - size_lambda * size)
        return _count_pipeline(credit, eps)
    if arm == "mwrl":
        credit = coverage_l2o_exact(masks, success, nu_p)
        return credit / (np.abs(credit).mean() + eps)
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
