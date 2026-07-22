"""The verifier: a pure sufficiency oracle ``s_c(S)`` computed as an exact closed-world subset test.

``s_c(S)`` opens the dimensions in ``S`` (the rest held at the baseline) and returns 1 if and
only if some tabulated assignment whose deviation set is contained in ``S`` clears the yield
threshold; assignments absent from the table count as failures. No Bayesian-optimization loop
runs here: on the complete closed-world panel the subset test equals the converged verdict of
the GP-EI inner search in ``bo.py`` (verified exactly, mask for mask, before the direct test
was adopted), which remains the wet-lab instantiation of the same contract. The verifier
reveals sufficiency and nothing else: it never consults the antichain, which is derived from
the data for evaluation only. Minimality is left to the MWRL coverage credit and measured post
hoc by single-dimension removal (also sufficiency queries).
"""

from __future__ import annotations

import numpy as np

from mwrl_suzuki.bo import BOInnerOptimizer
from mwrl_suzuki.data import ChemTask


def _is_subset(a: int, b: int) -> bool:
    return (a | b) == b


class ChemVerifier:
    def __init__(
        self,
        task: ChemTask,
        *,
        allowed: int | None = None,
        base_seed: int = 0,
        bo: BOInnerOptimizer | None = None,
    ) -> None:
        self.name = task.task_id
        self.base_seed = int(base_seed)
        self.n_comp = task.n_dims
        self.n_components = task.n_dims  # core Verifier protocol alias
        self.allowed = allowed if allowed is not None else (1 << task.n_dims) - 1
        self.attr_features = np.asarray(task.fingerprint, dtype=np.float32)
        self.antichain = list(task.antichain)          # ground-truth M(c); evaluation only
        self._yield_by_mask = dict(task.yield_by_mask)
        self.max_yield = float(task.max_yield)
        self._threshold = float(task.threshold)
        self._dim_values = task.dim_values
        self._baseline = task.baseline
        self._yield_table = task.yield_table
        self._bo = bo or BOInnerOptimizer()
        self.forward_calls = 0
        self._s0_cache: dict[int, bool] = {}

    def _yield_lookup(self, assignment: tuple[str, ...]) -> float:
        return self._yield_table.get(assignment, 0.0)   # unmeasured condition = a failed query (closed world)

    def s0(self, mask: int) -> bool:
        """Sufficiency of subspace ``mask``: the converged verdict of the inner search.

        A condition assignment is reachable in the subspace iff its deviation set is a subset
        of ``mask``, so on the closed-world table the converged inner search returns 1 exactly
        when some measured deviation set inside ``mask`` clears the threshold. This subset test
        is that converged verdict computed in O(measured rows); ``BOInnerOptimizer`` realizes
        the same search query-by-query (kept for open-world oracles) and provably returns the
        same bit on a closed-world table."""
        mask &= self.allowed
        cached = self._s0_cache.get(mask)
        if cached is not None:
            return cached
        self.forward_calls += 1
        ok = any(_is_subset(m, mask) and y >= self._threshold
                 for m, y in self._yield_by_mask.items())
        self._s0_cache[mask] = ok
        return ok

    def s0_batch(self, masks: list[int]) -> list[bool]:
        return [self.s0(m) for m in masks]

    def s_c(self, mask: int) -> bool:
        return self.s0(mask)

    def s_exists(self, mask: int) -> tuple[bool, int]:
        """(sufficient, witness). The witness is the raw opened set: the verifier does not prune to
        a minimal set, because minimality is the method's job, not the oracle's. Under the coverage
        credit the up-set measure rewards the policy for terminating on minimal witnesses."""
        mask &= self.allowed
        return (self.s0(mask), mask)

    def is_witness(self, mask: int) -> bool:
        return self.s0(mask)

    def faithfulness(self, mask: int) -> float:
        """Fraction of the task's best measured yield reachable while deviating only within mask."""
        mask &= self.allowed
        best = max((y for m, y in self._yield_by_mask.items() if _is_subset(m, mask)), default=0.0)
        return best / self.max_yield if self.max_yield > 0 else 0.0

    def ground_truth_antichain(self) -> list[int]:
        return list(self.antichain)
