"""Bayesian-optimization inner optimizer (GP + expected improvement).

Adapted from a reference Bayesian-optimization implementation so this package is
self-contained. This module is the validated reference implementation of the verifier's
wet-lab contract, not the runtime
path: the shipped verifier computes the identical verdict by an exact closed-world subset test
(agreement checked mask for mask before the swap). Given the opened dimensions (a subspace),
it searches over their categorical value-combinations, holding the unopened dimensions at the
baseline, querying yields from the reaction table. It returns as soon as a queried condition
clears the threshold, and otherwise runs to convergence over the whole subspace (no query cap),
so on a full-factorial table the sufficiency verdict is exact.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Assignment = tuple[str, ...]


@dataclass(frozen=True)
class BOSettings:
    initial_random_points: int = 4
    length_scale: float = 0.7
    signal_variance: float = 1.0
    noise_variance: float = 1.0e-5
    exploration: float = 0.01
    gp_budget: int = 24  # refit the GP for at most this many acquisitions, then scan the remaining
    #                      candidates directly. The GP only guides the SEARCH for a sufficient point;
    #                      proving insufficiency needs a full scan regardless, and refitting an O(t^3)
    #                      GP per query makes an unbounded scan O(C^4) in the subspace size C -- which
    #                      hangs on large insufficient subspaces. The verdict (a full-factorial table
    #                      stays exact) and the count of conditions queried are unchanged.


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


class BOInnerOptimizer:
    """GP-EI search over the opened dimensions' categorical values, holding others at baseline."""

    def __init__(self, settings: BOSettings | None = None) -> None:
        self.s = settings or BOSettings()

    def sufficient(
        self,
        active: Sequence[int],
        dim_values: Sequence[Sequence[str]],
        baseline: Sequence[str],
        yield_lookup: Callable[[Assignment], float],
        threshold: float,
        rng: np.random.Generator,
    ) -> bool:
        active = list(active)
        combos = list(itertools.product(*[range(len(dim_values[d])) for d in active])) if active else [()]

        def assignment(combo: tuple[int, ...]) -> Assignment:
            vals = list(baseline)
            for d, vi in zip(active, combo, strict=True):
                vals[d] = dim_values[d][vi]
            return tuple(vals)

        candidates = [assignment(c) for c in combos]
        obs: dict[int, float] = {}  # candidate index -> observed yield

        def query(idx: int) -> float:
            y = float(yield_lookup(candidates[idx]))
            obs[idx] = y
            return y

        order = list(range(len(candidates)))
        rng.shuffle(order)
        for idx in order[: max(1, self.s.initial_random_points)]:
            if query(idx) >= threshold:
                return True

        remaining = [i for i in order if i not in obs]
        while remaining:
            if 2 <= len(obs) < self.s.gp_budget:
                queried = list(obs)
                xt = self._encode([candidates[i] for i in queried], active, dim_values)
                yt = np.asarray([obs[i] for i in queried], dtype=np.float64)
                xc = self._encode([candidates[i] for i in remaining], active, dim_values)
                try:
                    ei = self._expected_improvement(xt, yt, xc)
                    pick = remaining[int(np.argmax(ei))]
                except np.linalg.LinAlgError:
                    pick = remaining[0]
            else:
                pick = remaining[0]
            remaining.remove(pick)
            if query(pick) >= threshold:
                return True
        return False

    def _encode(
        self, conditions: Sequence[Assignment], active: Sequence[int], dim_values: Sequence[Sequence[str]]
    ) -> np.ndarray:
        rows: list[list[float]] = []
        for cond in conditions:
            feats: list[float] = []
            for d in active:
                feats.extend(1.0 if cond[d] == level else 0.0 for level in dim_values[d])
            rows.append(feats if feats else [0.0])
        return np.asarray(rows, dtype=np.float64)

    def _expected_improvement(self, xt: np.ndarray, yt: np.ndarray, xc: np.ndarray) -> np.ndarray:
        ym, ys = float(yt.mean()), float(yt.std())
        scale = ys if ys > 1.0e-8 else 1.0
        yn = (yt - ym) / scale
        k = self._rbf(xt, xt) + np.eye(len(xt)) * self.s.noise_variance
        chol = np.linalg.cholesky(k + np.eye(len(xt)) * 1.0e-9)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yn))
        kc = self._rbf(xc, xt)
        mu = kc @ alpha
        v = np.linalg.solve(chol, kc.T)
        var = self.s.signal_variance - np.sum(v * v, axis=0)
        sd = np.sqrt(np.clip(var, 1.0e-12, None))
        best = float(np.max(yn))
        imp = mu - best - self.s.exploration
        z = imp / sd
        return imp * _normal_cdf(z) + sd * _normal_pdf(z)

    def _rbf(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        sq = np.sum(diff * diff, axis=-1)
        return self.s.signal_variance * np.exp(-0.5 * sq / (self.s.length_scale**2))
