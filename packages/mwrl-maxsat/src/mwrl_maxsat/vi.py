"""Executed value iteration on the MaxSAT subset lattice.

Two Bellman planners are actually run, not analytically asserted:

  * Scalar VI  -- terminal payoff ``s(S) - lambda |S|`` (or plain ``s(S)``). Backward
    induction over (mask, step); time-indexed decode with a payoff == value assertion.
    Repeated episodes return the same optimal witness, so its realized antichain recall
    is 1/|M| by execution.
  * Minimal-witness VI -- the exact budgeted planner on the augmented state (F, b).
    The coverage of every archive F subseteq M is obtained at once by a witness-incidence
    subset-sum (zeta) transform, backward induction over budgets executes
    V*(F,b) = max(coverage(F), max_m V*(F+m, b-1)), and the decoded archive's realized
    coverage is asserted equal to the Bellman value.

Verification: (i) the episode Bellman value at the empty set is checked against
brute-force enumeration of every trajectory on a small instance; (ii) the budgeted
optima are asserted against brute-force enumeration over all C(|M|, b) archives at
small budgets on every instance; (iii) every decode asserts realized payoff equals its
initial Bellman value.

    python -m mwrl_maxsat.vi          # 8 paper instances + verification, saves JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from mwrl_maxsat.instance import MaxSatInstance, generate_with_antichain, is_witness

# ---------------------------------------------------------------------------------------
# exact lattice machinery (n <= ~20: the small enumerable regime)
# ---------------------------------------------------------------------------------------


def _witness_vector(instance: MaxSatInstance) -> np.ndarray:
    n = instance.n_vars
    suff = np.zeros(1 << n, dtype=bool)
    for mask in range(1 << n):
        suff[mask] = is_witness(instance, mask)      # the black-box oracle, queried per state
    return suff


def _weights(n: int, p: float) -> np.ndarray:
    sizes = np.array([int(m).bit_count() for m in range(1 << n)], dtype=np.int16)
    return np.power(p, sizes) * np.power(1.0 - p, n - sizes)


def _superset_sum(values: np.ndarray, n: int) -> np.ndarray:
    """Zeta transform: out[S] = sum_{T >= S} values[T], O(n 2^n)."""
    out = values.astype(np.float64).copy()
    for i in range(n):
        bit = 1 << i
        idx = np.arange(1 << n)
        lo = (idx & bit) == 0
        out[idx[lo]] += out[idx[lo] | bit]
    return out


def lattice_vi(n: int, payoff: np.ndarray, horizon: int) -> tuple[float, int]:
    """Executed backward induction over states (mask, t), actions {add i, stop}.

    ``payoff[mask]`` is the terminal reward of stopping at ``mask``. Returns the Bellman
    value at (empty set, full horizon) and a decoded terminal witness. Every
    time-indexed value table is stored and the decode consults the table matching the
    steps that remain, so a promise that needs more depth than remains can never be
    followed; the decoded terminal's payoff is asserted equal to the Bellman value.
    """
    n_masks = 1 << n
    idx = np.arange(n_masks)
    add_targets = [idx | (1 << i) for i in range(n)]
    tables = [payoff.copy()]                              # tables[r] = value with r steps left
    for _ in range(horizon):
        v_prev = tables[-1]
        v_add = np.full(n_masks, -np.inf)
        for i in range(n):
            tgt = v_prev[add_targets[i]]
            keep = (idx & (1 << i)) == 0                  # only genuine additions
            v_add[keep] = np.maximum(v_add[keep], tgt[keep])
        tables.append(np.maximum(payoff, v_add))          # stop now, or add and continue

    # decode against the remaining-steps table (deterministic tie-break: lowest index)
    mask, value = 0, float(tables[horizon][0])
    remaining = horizon
    while remaining > 0 and payoff[mask] < value - 1e-12:
        for i in range(n):
            if not (mask >> i) & 1 and tables[remaining - 1][mask | (1 << i)] >= value - 1e-12:
                mask |= 1 << i
                remaining -= 1
                break
        else:
            raise AssertionError("decode found no action attaining the Bellman value")
    assert abs(payoff[mask] - value) < 1e-9, (
        f"decoded payoff {payoff[mask]} != Bellman value {value}")
    return value, mask


def _brute_force_value(n: int, payoff: np.ndarray, horizon: int, mask: int = 0) -> float:
    """Enumerate every trajectory (verification only; exponential)."""
    best = float(payoff[mask])
    if horizon == 0:
        return best
    for i in range(n):
        if not (mask >> i) & 1:
            best = max(best, _brute_force_value(n, payoff, horizon - 1, mask | (1 << i)))
    return best


# ---------------------------------------------------------------------------------------
# the two executed planners
# ---------------------------------------------------------------------------------------


def scalar_vi(instance: MaxSatInstance, *, lam: float, horizon: int, budget: int,
              antichain: list[int]) -> dict:
    """Scalar planner, executed for ``budget`` episodes; payoff = s(S) - lam |S|."""
    n = instance.n_vars
    suff = _witness_vector(instance)
    sizes = np.array([int(m).bit_count() for m in range(1 << n)], dtype=np.float64)
    payoff = suff.astype(np.float64) - lam * sizes
    ac = set(antichain)
    found: set[int] = set()
    witnesses = []
    for _ in range(budget):
        value, w = lattice_vi(n, payoff, horizon)
        witnesses.append(w)
        if w in ac:
            found.add(w)
    return {
        "lambda": lam,
        "value": value,
        "distinct_witnesses": len(set(witnesses)),
        "found_minimal": len(found),
        "completeness": len(found) / len(ac),
        "witness_size": int(witnesses[0]).bit_count(),
        "witness_is_minimal": witnesses[0] in ac,
    }


def minimal_witness_vi(instance: MaxSatInstance, *, p: float, horizon: int,
                       antichain: list[int]) -> dict:
    """The exact budgeted planner, executed on the augmented state (F, b).

    Optimal episodes terminate at minimal witnesses and every witness is reachable
    within the horizon, so transitions add one element of M to the archive F. The
    coverage of every archive is computed at once: each lattice point T carries a
    witness-incidence mask (which minimal witnesses T contains), the nu-mass of each
    incidence class is accumulated, and a subset-sum (zeta) transform over the
    |M|-cube yields uncovered(A) for all 2^|M| archives in O(|M| 2^|M|). Backward
    induction over budgets then executes V*(F,b) = max(coverage(F), max_m V*(F+m,b-1));
    the decoded archive's realized coverage is asserted equal to the Bellman value, and
    the per-budget optima are asserted against brute-force subset enumeration at small
    budgets.
    """
    n = instance.n_vars
    ac = list(antichain)
    L = len(ac)
    assert max(int(m).bit_count() for m in ac) <= horizon, "witness beyond the horizon"
    w_lattice = _weights(n, p)
    lattice = np.arange(1 << n)

    # witness incidence per lattice point, and the nu-mass of each incidence class
    incidence = np.zeros(1 << n, dtype=np.int64)
    for j, m in enumerate(ac):
        incidence[(lattice & m) == m] |= 1 << j
    g = np.zeros(1 << L)
    np.add.at(g, incidence, w_lattice)

    # uncovered(A) = sum of g over incidence classes disjoint from A: zeta then complement
    zeta = g.copy()
    archive_idx = np.arange(1 << L)
    for j in range(L):
        bit = 1 << j
        has = (archive_idx & bit) != 0
        zeta[archive_idx[has]] += zeta[archive_idx[has] ^ bit]
    full = (1 << L) - 1
    coverage = 1.0 - zeta[(~archive_idx) & full]
    ceiling = float(coverage[full])                       # nu(upset(M))

    # executed backward induction over the augmented state (F, b)
    tables = [coverage.copy()]                            # tables[b][F] = V*(F, b)
    for _b in range(L):
        v_prev = tables[-1]
        v_next = coverage.copy()
        for j in range(L):
            bit = 1 << j
            free = (archive_idx & bit) == 0
            np.maximum(v_next, np.where(free, v_prev[archive_idx | bit], -np.inf),
                       out=v_next)
        tables.append(v_next)
    budget_values = [float(tables[b][0]) for b in range(L + 1)]     # V*(empty, b)

    # verification: brute-force subset enumeration at small budgets
    budget_check = 0.0
    for b in range(1, min(L, 4) + 1):
        brute = exact_budget_optimum(instance, p=p, antichain=ac, budget=b)
        budget_check = max(budget_check, abs(brute - budget_values[b]))
    assert budget_check < 1e-9, f"budgeted optimum mismatch: {budget_check}"

    # decode the full-budget optimal archive against the remaining-budget tables
    archive_mask, value = 0, budget_values[L]
    remaining = L
    while remaining > 0 and coverage[archive_mask] < value - 1e-12:
        for j in range(L):
            bit = 1 << j
            if not archive_mask & bit and tables[remaining - 1][archive_mask | bit] >= value - 1e-12:
                archive_mask |= bit
                remaining -= 1
                break
        else:
            raise AssertionError("decode found no addition attaining the Bellman value")
    assert abs(coverage[archive_mask] - value) < 1e-9
    archive = [ac[j] for j in range(L) if archive_mask & (1 << j)]

    return {
        "L": L,
        "recall_curve": [b / L for b in range(1, L + 1)],  # optimal b-archives are b-subsets of M
        "coverage_curve": budget_values[1:],
        "budget_values": budget_values,
        "budget_check": float(budget_check),
        "final_recall": len(archive) / L,
        "final_coverage": float(coverage[archive_mask]),
        "coverage_ceiling": ceiling,
        "all_terminals_minimal": bool(all(m in set(ac) for m in archive)),
        "found": len(archive),
        "median_size": float(np.median([int(m).bit_count() for m in archive])),
    }


def exact_budget_optimum(instance: MaxSatInstance, *, p: float, antichain: list[int],
                         budget: int) -> float:
    """Exact optimum of nu(union of upsets) over all C(|M|, b) archives (verification)."""
    n = instance.n_vars
    w_lattice = _weights(n, p)
    idx = np.arange(1 << n)
    best = 0.0
    for combo in combinations(antichain, budget):
        covered = np.zeros(1 << n, dtype=bool)
        for m in combo:
            covered |= (idx & m) == m
        best = max(best, float(np.sum(w_lattice[covered])))
    return best


# ---------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--n-vars", type=int, default=14)
    ap.add_argument("--n-clauses", type=int, default=10)
    ap.add_argument("--clause-len", type=int, default=3)
    ap.add_argument("--min-modes", type=int, default=6)
    ap.add_argument("--max-modes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--p", type=float, default=0.7)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--verify-exact-max-L", type=int, default=12)
    ap.add_argument("--output", default="runs/maxsat_vi")
    args = ap.parse_args()

    # --- verification 1: Bellman value equals brute-force trajectory enumeration ----------
    tiny, tiny_ac = generate_with_antichain(8, 6, 3, min_modes=2, max_modes=6,
                                            rng=np.random.default_rng(7))
    tiny_suff = _witness_vector(tiny)
    tiny_payoff = np.where(tiny_suff, _superset_sum(np.where(tiny_suff, _weights(8, args.p), 0.0), 8), 0.0)
    bellman, _ = lattice_vi(8, tiny_payoff, 6)
    brute = _brute_force_value(8, tiny_payoff, 6)
    assert abs(bellman - brute) < 1e-12, (bellman, brute)
    print(f"[verify] Bellman == brute force on n=8: {bellman:.6f} == {brute:.6f}", flush=True)

    results = []
    for i in range(args.instances):
        inst, ac = generate_with_antichain(args.n_vars, args.n_clauses, args.clause_len,
                                           min_modes=args.min_modes, max_modes=args.max_modes,
                                           rng=np.random.default_rng(1000 + i))
        L = len(ac)
        scalar_pen = scalar_vi(inst, lam=args.lam, horizon=args.horizon, budget=L, antichain=ac)
        scalar_pure = scalar_vi(inst, lam=0.0, horizon=args.horizon, budget=L, antichain=ac)
        mw = minimal_witness_vi(inst, p=args.p, horizon=args.horizon, antichain=ac)
        results.append({"instance": i, "L": L, "scalar_penalized": scalar_pen,
                        "scalar_pure": scalar_pure, "minimal_witness": mw})
        print(f"[inst {i}] L={L:2d}  scalarVI(lam={args.lam}): compl={scalar_pen['completeness']:.3f} "
              f"minimal={scalar_pen['witness_is_minimal']}  scalarVI(0): |S|={scalar_pure['witness_size']} "
              f"minimal={scalar_pure['witness_is_minimal']}  budgeted MW-VI: recall={mw['final_recall']:.2f} "
              f"cov={mw['final_coverage']:.3f}/{mw['coverage_ceiling']:.3f} "
              f"all-minimal={mw['all_terminals_minimal']}  "
              f"budget-check={mw['budget_check']:.1e}", flush=True)

    agg = {
        "scalar_completeness": float(np.mean([r["scalar_penalized"]["completeness"] for r in results])),
        "scalar_pure_size": float(np.mean([r["scalar_pure"]["witness_size"] for r in results])),
        "scalar_pure_minimal_rate": float(np.mean([r["scalar_pure"]["witness_is_minimal"] for r in results])),
        "mw_final_recall": float(np.mean([r["minimal_witness"]["final_recall"] for r in results])),
        "mw_coverage_ratio": float(np.mean([r["minimal_witness"]["final_coverage"]
                                            / r["minimal_witness"]["coverage_ceiling"] for r in results])),
        "max_budget_check": float(max(r["minimal_witness"]["budget_check"] for r in results)),
        "all_terminals_minimal": bool(all(r["minimal_witness"]["all_terminals_minimal"] for r in results)),
    }
    print("\n[aggregate]", json.dumps(agg, indent=2), flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "executed_vi.json").write_text(json.dumps({"config": vars(args), "results": results,
                                                      "aggregate": agg}, indent=2))
    print(f"[saved] {out / 'executed_vi.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
