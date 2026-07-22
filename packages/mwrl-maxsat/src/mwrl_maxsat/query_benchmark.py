"""Query-accounted black-box deletion baseline for the MaxSAT benchmark.

The baseline verifies the full variable set once, then repeatedly greedily deletes
variables in a fresh random order.  Every call to the sufficiency predicate is
counted.  Under exact monotonicity each completed pass returns a minimal witness.

Run the paper-instance benchmark with

    python -m mwrl_maxsat.query_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mwrl_maxsat.instance import MaxSatInstance, generate_with_antichain, is_witness

DEFAULT_BUDGETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 57600)


@dataclass
class CountingOracle:
    """Count total and distinct sufficiency queries without caching responses."""

    instance: MaxSatInstance
    requests: int = 0
    queried_masks: set[int] = field(default_factory=set)

    def __call__(self, mask: int) -> bool:
        self.requests += 1
        self.queried_masks.add(mask)
        return is_witness(self.instance, mask)


def randomized_greedy_delete(
    oracle: CountingOracle,
    seed_mask: int,
    rng: np.random.Generator,
) -> int:
    """Return one deletion-minimal witness using exactly ``n_vars`` queries.

    ``seed_mask`` is the full set and is verified once outside this function.  Every
    variable is still present when its turn is reached, so one predicate query is
    made for each variable.
    """
    current = seed_mask
    for variable in rng.permutation(oracle.instance.n_vars):
        candidate = current & ~(1 << int(variable))
        if oracle(candidate):
            current = candidate
    return current


def _coverage_lookup(antichain: list[int], n_vars: int, p: float) -> tuple[dict[int, np.ndarray], float]:
    """Precompute exact product-measure up-sets on the finite lattice."""
    states = np.arange(1 << n_vars, dtype=np.uint64)
    sizes = np.fromiter((int(x).bit_count() for x in states), dtype=np.int16)
    weights = np.power(p, sizes) * np.power(1.0 - p, n_vars - sizes)
    upsets = {mask: (states & np.uint64(mask)) == np.uint64(mask) for mask in antichain}
    covered = np.zeros(states.shape[0], dtype=bool)
    for upset in upsets.values():
        covered |= upset
    return upsets, float(weights[covered].sum())


def _run_one(
    instance: MaxSatInstance,
    antichain: list[int],
    *,
    instance_id: int,
    repeat: int,
    budgets: tuple[int, ...],
    p: float,
    seed: int,
) -> tuple[list[dict], dict]:
    truth = set(antichain)
    full = (1 << instance.n_vars) - 1
    oracle = CountingOracle(instance)
    if not oracle(full):
        raise RuntimeError("the full set is not sufficient")

    rng = np.random.default_rng([seed, instance_id, repeat])
    upsets, full_coverage = _coverage_lookup(antichain, instance.n_vars, p)
    states = np.arange(1 << instance.n_vars, dtype=np.uint64)
    sizes = np.fromiter((int(x).bit_count() for x in states), dtype=np.int16)
    weights = np.power(p, sizes) * np.power(1.0 - p, instance.n_vars - sizes)
    covered = np.zeros(states.shape[0], dtype=bool)
    found: set[int] = set()
    restarts = 0
    milestones: dict[str, int | None] = {"q50": None, "q90": None, "q100": None}
    records: list[dict] = []

    for budget in budgets:
        while oracle.requests + instance.n_vars <= budget:
            witness = randomized_greedy_delete(oracle, full, rng)
            restarts += 1
            if witness not in truth:
                raise AssertionError("greedy deletion did not return a ground-truth minimal witness")
            if witness not in found:
                found.add(witness)
                covered |= upsets[witness]
                recall = len(found) / len(truth)
                for name, target in (("q50", 0.5), ("q90", 0.9), ("q100", 1.0)):
                    if milestones[name] is None and recall >= target:
                        milestones[name] = oracle.requests

        coverage = float(weights[covered].sum()) / full_coverage
        records.append(
            {
                "instance": instance_id,
                "repeat": repeat,
                "budget": budget,
                "queries": oracle.requests,
                "distinct_queries": len(oracle.queried_masks),
                "restarts": restarts,
                "antichain_size": len(truth),
                "found": len(found),
                "recall": len(found) / len(truth),
                "normalized_upset_coverage": coverage,
                "full_recovery": len(found) == len(truth),
                "raw_seed_born_minimal": full in truth,
                "post_deletion_precision": 1.0,
            }
        )
    return records, milestones


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return float(array.mean()), std


def _aggregate(records: list[dict], budgets: tuple[int, ...], milestones: list[dict]) -> dict:
    by_budget: list[dict] = []
    for budget in budgets:
        rows = [row for row in records if row["budget"] == budget]
        recall_mean, recall_std = _mean_std([row["recall"] for row in rows])
        coverage_mean, coverage_std = _mean_std([row["normalized_upset_coverage"] for row in rows])
        found_mean, found_std = _mean_std([row["found"] for row in rows])
        by_budget.append(
            {
                "budget": budget,
                "runs": len(rows),
                "mean_queries_used": float(np.mean([row["queries"] for row in rows])),
                "mean_distinct_queries": float(np.mean([row["distinct_queries"] for row in rows])),
                "mean_found": found_mean,
                "std_found": found_std,
                "mean_recall": recall_mean,
                "std_recall": recall_std,
                "mean_normalized_upset_coverage": coverage_mean,
                "std_normalized_upset_coverage": coverage_std,
                "full_recovery_rate": float(np.mean([row["full_recovery"] for row in rows])),
            }
        )

    milestone_summary: dict[str, dict] = {}
    for name in ("q50", "q90", "q100"):
        observed = [row[name] for row in milestones if row[name] is not None]
        milestone_summary[name] = {
            "solved_fraction": len(observed) / len(milestones),
            "mean_queries_when_reached": float(np.mean(observed)) if observed else None,
            "median_queries_when_reached": float(np.median(observed)) if observed else None,
        }
    return {"by_budget": by_budget, "milestones": milestone_summary}


def _parse_budgets(raw: str) -> tuple[int, ...]:
    budgets = tuple(sorted({int(value) for value in raw.split(",")}))
    if not budgets or budgets[0] < 1:
        raise argparse.ArgumentTypeError("budgets must be positive integers")
    return budgets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--n-vars", type=int, default=14)
    parser.add_argument("--n-clauses", type=int, default=10)
    parser.add_argument("--clause-len", type=int, default=3)
    parser.add_argument("--min-modes", type=int, default=6)
    parser.add_argument("--max-modes", type=int, default=20)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--budgets",
        type=_parse_budgets,
        default=DEFAULT_BUDGETS,
        help="comma-separated total verifier-query budgets",
    )
    parser.add_argument("--output", default="runs/maxsat_validations/randomized_deletion.json")
    args = parser.parse_args()
    if not 0.0 < args.p < 1.0:
        parser.error("--p must lie strictly between zero and one")

    instances: list[tuple[MaxSatInstance, list[int]]] = []
    for instance_id in range(args.instances):
        instances.append(
            generate_with_antichain(
                args.n_vars,
                args.n_clauses,
                args.clause_len,
                min_modes=args.min_modes,
                max_modes=args.max_modes,
                rng=np.random.default_rng(1000 + instance_id),
            )
        )

    records: list[dict] = []
    milestones: list[dict] = []
    for instance_id, (instance, antichain) in enumerate(instances):
        for repeat in range(args.repeats):
            run_records, run_milestones = _run_one(
                instance,
                antichain,
                instance_id=instance_id,
                repeat=repeat,
                budgets=args.budgets,
                p=args.p,
                seed=args.seed,
            )
            records.extend(run_records)
            milestones.append({"instance": instance_id, "repeat": repeat, **run_milestones})

    config = {
        "instances": args.instances,
        "repeats": args.repeats,
        "n_vars": args.n_vars,
        "n_clauses": args.n_clauses,
        "clause_len": args.clause_len,
        "min_modes": args.min_modes,
        "max_modes": args.max_modes,
        "p": args.p,
        "seed": args.seed,
        "budgets": args.budgets,
        "query_accounting": "one full-set verification plus n_vars calls per restart",
        "paper_training_budget": 150 * 8 * 48,
    }
    payload = {
        "config": config,
        "antichain_sizes": [len(antichain) for _, antichain in instances],
        "aggregate": _aggregate(records, args.budgets, milestones),
        "milestones": milestones,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))

    print(f"antichain sizes: {payload['antichain_sizes']}")
    print(f"{'budget':>8} {'recall':>9} {'coverage':>10} {'full':>8}")
    for row in payload["aggregate"]["by_budget"]:
        print(
            f"{row['budget']:8d} {row['mean_recall']:9.3f} "
            f"{row['mean_normalized_upset_coverage']:10.3f} "
            f"{row['full_recovery_rate']:8.3f}"
        )
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
