"""Monotone MaxSAT instances, the sufficiency predicate, and exact minimal witnesses.

A *monotone* instance has clauses of positive literals only; a variable subset ``S``
(the variables set true, a bitmask) satisfies a clause iff it intersects it. The
sufficiency predicate is

    s(S) = 1  iff  #{clauses satisfied by S} >= threshold,

which is monotone in ``S`` (adding a true variable never unsatisfies a clause). Its
minimal witnesses ``M`` are the prime implicants -- minimal subsets clearing the
threshold (minimal hitting sets when threshold = #clauses). For small ``n`` these
are enumerable exactly, giving ground truth to score antichain recovery against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mwrl.credit import _prune_to_antichain, popcount


@dataclass(frozen=True)
class MaxSatInstance:
    n_vars: int
    clauses: tuple[frozenset[int], ...]  # each clause = set of variable indices (monotone)
    threshold: int  # satisfy at least this many clauses

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)


def satisfied_count(instance: MaxSatInstance, mask: int) -> int:
    """Number of clauses hit by the variables set true in ``mask``."""
    return sum(1 for clause in instance.clauses
               if any((mask >> v) & 1 for v in clause))


def is_witness(instance: MaxSatInstance, mask: int) -> bool:
    """The monotone sufficiency predicate s(S)."""
    return satisfied_count(instance, mask) >= instance.threshold


def minimal_witnesses(instance: MaxSatInstance) -> list[int]:
    """Exact antichain M of minimal witnesses (brute force; small n only)."""
    witnesses = [m for m in range(1 << instance.n_vars) if is_witness(instance, m)]
    return sorted(_prune_to_antichain(witnesses), key=popcount)


def canonicalize(instance: MaxSatInstance, mask: int, *,
                 order: list[int] | None = None) -> int:
    """Greedy single-variable prune to a minimal witness (exact under monotone s).

    Requires ``is_witness(mask)``. Under a monotone predicate one pass in any order
    reaches a minimal witness (PDF 7.2's Phi). ``order`` should be a *deterministic*
    function of the mask so Phi is a function S -> M(c): the antichain diversity then
    has to come from the POLICY proposing different S. A fresh-random order per call
    would instead make Phi a stochastic antichain enumerator that any policy inherits
    -- the brute-force anti-pattern, which trivializes count on low-dim instances.
    """
    order = order if order is not None else list(range(instance.n_vars))
    current = mask
    for v in order:
        bit = 1 << v
        if (current & bit) and is_witness(instance, current & ~bit):
            current &= ~bit
    return current


def generate_monotone(
    n_vars: int,
    n_clauses: int,
    clause_len: int,
    *,
    threshold: int | None = None,
    rng: np.random.Generator,
) -> MaxSatInstance:
    """Random monotone k-CNF. ``threshold=None`` means satisfy ALL clauses
    (minimal witnesses = minimal hitting sets)."""
    clauses: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()
    while len(clauses) < n_clauses:
        clause = frozenset(int(v) for v in rng.choice(n_vars, size=clause_len, replace=False))
        if clause not in seen:
            seen.add(clause)
            clauses.append(clause)
    return MaxSatInstance(
        n_vars=n_vars,
        clauses=tuple(clauses),
        threshold=n_clauses if threshold is None else threshold,
    )


def generate_with_antichain(
    n_vars: int,
    n_clauses: int,
    clause_len: int,
    *,
    min_modes: int,
    max_modes: int,
    threshold: int | None = None,
    rng: np.random.Generator,
    tries: int = 400,
) -> tuple[MaxSatInstance, list[int]]:
    """Rejection-sample an instance whose exact antichain size is in a target band,
    so we can dial the *multimodality* the antichain recovery hinges on."""
    for _ in range(tries):
        inst = generate_monotone(n_vars, n_clauses, clause_len, threshold=threshold, rng=rng)
        witnesses = minimal_witnesses(inst)
        if min_modes <= len(witnesses) <= max_modes:
            return inst, witnesses
    raise RuntimeError(
        f"no instance with antichain size in [{min_modes},{max_modes}] after {tries} tries"
    )
