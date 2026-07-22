"""Lattice-valued leave-one-out credit for grouped rollouts.

Phase 1 uses the deduplicated mode-count objective (design section 4.4): after
environment-side canonicalization every recorded witness is a minimal element, so
credit is the exact leave-one-out marginal of the count valuation -- a 0/1 signal
that is non-zero exactly on the within-group minimal antichain. Phase 2's coverage
objective (design section 4.3) is provided too but not yet wired into training.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def popcount(mask: int) -> int:
    return bin(mask).count("1")


def count_advantages(canon_masks: list[int], succ: list[bool]) -> list[float]:
    """A_i = s_i * 1[canon_i is not duplicated by another success] (design section 4.4).

    Duplicated modes each receive zero credit: removing one copy does not lower the
    certified count because the other copy still certifies that mode.
    """
    k = len(canon_masks)
    advantages = [0.0] * k
    for i in range(k):
        if not succ[i]:
            continue
        if all(
            not (succ[j] and canon_masks[j] == canon_masks[i])
            for j in range(k)
            if j != i
        ):
            advantages[i] = 1.0
    return advantages


def _mc_union_measure(masks: list[int], p: float, n_samples: int, rng) -> float:
    """Monte-Carlo estimate of nu(union of up-sets) under the product(p) measure:
    nu(union) = P_{T~nu}(T contains some S_i), by sampling T (each coordinate included
    independently with probability p) and testing containment. Only coordinates that appear
    in the masks affect containment, so we sample just those -- O(n_samples * masks * coords),
    no exponential in the antichain size. This is theory 4.4's MC prescription for large |D_i|."""
    universe = 0
    for m in masks:
        universe |= m
    if universe == 0:
        return 0.0
    coords = [c for c in range(universe.bit_length()) if (universe >> c) & 1]
    idx = {c: j for j, c in enumerate(coords)}
    membership = np.zeros((len(masks), len(coords)), dtype=bool)
    for i, m in enumerate(masks):
        for c in range(m.bit_length()):
            if (m >> c) & 1:
                membership[i, idx[c]] = True
    t = rng.random((n_samples, len(coords))) < p                    # sampled sets T
    missing = (membership[None, :, :] & ~t[:, None, :]).any(axis=2)  # S_i not a subset of T
    return float((~missing).any(axis=1).mean())                     # P(some S_i subset of T)


def union_upset_measure(success_masks: list[int], nu_upset=None, *, mc_p: float | None = None,
                        mc_samples: int = 0, mc_threshold: int = 12, rng=None) -> float:
    """nu(union of up-sets over the given masks). Exact antichain-pruned inclusion-exclusion
    when the within-group antichain is small; a Monte-Carlo estimate under the product(mc_p)
    measure once it exceeds ``mc_threshold`` (the exact form is O(2^antichain), so a large group
    on a multi-mode instance is otherwise intractable). MC applies to the product measures
    (uniform p=1/2, geometric p); layer has no product form so it stays exact (mc_p=None)."""
    if nu_upset is None:
        nu_upset = nu_upset_layer
    antichain = _prune_to_antichain(list(success_masks))
    if not antichain:
        return 0.0
    if mc_p is not None and mc_samples > 0 and len(antichain) > mc_threshold:
        return _mc_union_measure(antichain, mc_p, mc_samples, rng or np.random.default_rng())
    total = 0.0
    for r in range(1, len(antichain) + 1):
        for subset in combinations(antichain, r):
            union = 0
            for mask in subset:
                union |= mask
            total += (-1) ** (r + 1) * nu_upset(popcount(union))
    return total


def _mc_shared_l2o(masks: list[int], succ: list[bool], p: float, n_samples: int,
                    rng) -> list[float]:
    """Shared-sample O(NK) estimator of the leave-two-out coverage credit (appendix D).

    One batch of lattice samples T ~ product(p) serves every union value, so the
    differences are computed per sample and the union-level noise cancels exactly:
    the raw marginal of proposal i is the fraction of samples containing S_i and no
    other success, and the leave-two-out baseline is 1/(K-1) times the fraction of
    samples containing exactly one other success. Estimating each union with fresh
    samples and subtracting, by contrast, leaves O(sigma) noise on an O(marginal)
    difference and flips credit signs (see runs/maxsat_validations/mc_credit.json).
    """
    k = len(masks)
    universe = 0
    for m, s in zip(masks, succ, strict=True):
        if s:
            universe |= m
    if universe == 0:
        return [0.0] * k
    coords = [c for c in range(universe.bit_length()) if (universe >> c) & 1]
    idx = {c: j for j, c in enumerate(coords)}
    membership = np.zeros((k, len(coords)), dtype=bool)     # failed rows stay all-False
    for i, (m, s) in enumerate(zip(masks, succ, strict=True)):
        if not s:
            continue
        for c in range(m.bit_length()):
            if (m >> c) & 1:
                membership[i, idx[c]] = True
    t = rng.random((n_samples, len(coords))) < p                            # sampled sets T
    cover = ~((membership[None, :, :] & ~t[:, None, :]).any(axis=2))        # T contains S_i
    for i, s in enumerate(succ):                                            # failures never cover
        if not s:
            cover[:, i] = False
    count = cover.sum(axis=1)
    raw = (cover & (count[:, None] == 1)).mean(axis=0, dtype=np.float64)    # unique coverer
    if k > 1:
        baseline = ((count[:, None] - cover) == 1).mean(axis=0, dtype=np.float64) / (k - 1)
    else:
        baseline = np.zeros(k, dtype=np.float64)
    return [float(v) for v in raw - baseline]


def coverage_l2o_advantages(masks: list[int], succ: list[bool], *, nu_upset=None,
                             mc_p: float | None = None, mc_samples: int = 0,
                             mc_threshold: int = 12, rng=None) -> list[float]:
    """Coverage credit with the exact leave-two-out (unbiased) centering (theory section 7.1).

    A_i = V(U_K) - V(U_{-i}); the centered advantage subtracts a baseline built only from
    the other samples' region, c_i = V(U_{-i}) - mean_{j!=i} V(U_{-i-j}), which is independent
    of sample i (so the estimator stays unbiased despite the credits being coupled through U_K).

    The estimator is decided ONCE per group: exact inclusion-exclusion while the group's
    pruned success antichain is small, and the shared-sample Monte-Carlo estimator
    (``_mc_shared_l2o``) beyond ``mc_threshold``. All values within a group come from one
    method, and the MC path evaluates every union on one common sample batch; per-union
    fresh sampling is not admissible because differencing independent estimates destroys
    the credit's sign structure.
    """
    if nu_upset is None:
        nu_upset = nu_upset_layer
    k = len(masks)
    succ_indices = [i for i in range(k) if succ[i]]

    antichain = _prune_to_antichain([masks[j] for j in succ_indices])
    if mc_p is not None and mc_samples > 0 and len(antichain) > mc_threshold:
        return _mc_shared_l2o(masks, succ, mc_p, mc_samples,
                               rng or np.random.default_rng())

    cache: dict[frozenset[int], float] = {}

    def measure_excluding(excluded: frozenset[int]) -> float:
        key = excluded & frozenset(succ_indices)
        if key not in cache:
            cache[key] = union_upset_measure([masks[j] for j in succ_indices if j not in key],
                                             nu_upset)
        return cache[key]

    v_full = measure_excluding(frozenset())
    advantages = [0.0] * k
    for i in range(k):
        v_minus_i = measure_excluding(frozenset({i}))
        marginal = v_full - v_minus_i  # A_i (0 if i failed or dominated)
        others = [j for j in range(k) if j != i]
        if others:
            baseline = v_minus_i - sum(measure_excluding(frozenset({i, j})) for j in others) / len(others)
        else:
            baseline = 0.0
        advantages[i] = marginal - baseline
    return advantages


def nu_upset_uniform(size: int) -> float:
    """nu(up-set of U) under the uniform measure, |U| = size."""
    return 2.0 ** (-size)


def nu_upset_layer(size: int) -> float:
    """nu(up-set of U) under the layer-uniform measure (recommended), |U| = size."""
    return 1.0 / (size + 1)


def nu_upset_geometric(p: float):
    """nu(up-set) under the product measure with uniform inclusion prob ``p``: nu(size)=p**size.

    The one-parameter family that interpolates the two fixed measures: p=0.5 is the uniform
    2^-|S| (harsh shrink signal, crushes large-but-necessary), p->1 flattens toward no signal.
    Its per-element shrink preference nu(m)/nu(m+1)=1/p is the SAME at every size -- the unique
    scale-invariant choice, unlike the layer measure whose preference vanishes as ~1/|S|.
    """
    def nu(size: int) -> float:
        return p ** size
    return nu


def _prune_to_antichain(masks: list[int]) -> list[int]:
    out: list[int] = []
    for candidate in sorted(set(masks), key=popcount):
        if not any(kept & candidate == kept for kept in out):  # some kept subset of candidate
            out.append(candidate)
    return out


def coverage_advantages(
    masks: list[int],
    succ: list[bool],
    *,
    nu_upset=nu_upset_layer,
) -> list[float]:
    """A_i = s_i * nu(up(S_i) \\ union_{j!=i, succ} up(S_j)) (design section 4.3).

    Exact inclusion-exclusion over the within-group minimal antichain. Dominated
    successes get exactly zero by the support lemma.
    """
    k = len(masks)
    advantages = [0.0] * k
    for i in range(k):
        if not succ[i]:
            continue
        dominators = _prune_to_antichain([masks[j] for j in range(k) if j != i and succ[j]])
        if any((x & masks[i]) == x for x in dominators):  # some x subset of S_i: dominated
            continue
        total = nu_upset(popcount(masks[i]))
        for r in range(1, len(dominators) + 1):
            for subset in combinations(dominators, r):
                union = masks[i]
                for mask in subset:
                    union |= mask
                total += (-1) ** r * nu_upset(popcount(union))
        advantages[i] = total
    return advantages
