"""Controlled MaxSAT stress test with one exponentially rare deletion basin.

The planted antichain contains ``q`` singleton witnesses and one disjoint witness of
size ``m``.  It is represented exactly as a monotone CNF.  Randomized deletion from
the full set reaches the large witness only when all singleton variables precede all
large-witness variables in the deletion order, with probability ``1 / C(q+m, q)``.

This is a compact diagnostic cell for the proposed antichain-geometry phase diagram.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from mwrl.config import MWRLConfig

from mwrl_maxsat.env import MaxSatEnv
from mwrl_maxsat.instance import MaxSatInstance, minimal_witnesses
from mwrl_maxsat.learned_query_benchmark import DiscoveryRunner
from mwrl_maxsat.query_benchmark import CountingOracle, randomized_greedy_delete

DEFAULT_BUDGETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 57600)


def make_rare_basin_instance(singletons: int, rare_size: int) -> tuple[MaxSatInstance, list[int], int]:
    """Return a monotone CNF with singleton modes plus one disjoint large mode."""
    n_vars = singletons + rare_size
    common = frozenset(range(singletons))
    rare_variables = tuple(range(singletons, n_vars))
    clauses = tuple(common | {variable} for variable in rare_variables)
    instance = MaxSatInstance(n_vars=n_vars, clauses=clauses, threshold=len(clauses))
    rare_mask = sum(1 << variable for variable in rare_variables)
    antichain = [1 << variable for variable in range(singletons)] + [rare_mask]
    if set(minimal_witnesses(instance)) != set(antichain):
        raise AssertionError("planted CNF does not reproduce its target antichain")
    return instance, antichain, rare_mask


class RareDiscoveryRunner(DiscoveryRunner):
    def __init__(self, *args, rare_mask: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rare_mask = rare_mask

    def _archive(self, rolls) -> None:
        previous = len(self.records)
        super()._archive(rolls)
        for record in self.records[previous:]:
            record["rare_found"] = self.rare_mask in self.found


def _mwrl_worker(job: tuple[int, dict]) -> dict:
    seed, cfg = job
    torch.set_num_threads(1)
    instance, antichain, rare_mask = make_rare_basin_instance(cfg["singletons"], cfg["rare_size"])
    config = MWRLConfig(
        updates=cfg["updates"],
        task_batch_size=cfg["groups_per_update"],
        group_size=cfg["group_size"],
        learning_rate=3e-3,
        entropy_coef=0.02,
        hidden_dim=128,
        update_epochs=1,
        minibatch_size=256,
        clip_epsilon=0.2,
        max_grad_norm=1.0,
        epsilon=1e-6,
        witness_valuation="coverage",
        witness_center=True,
        witness_baseline="l2o",
        witness_normalize=True,
        witness_scale="mean",
        witness_nu="geometric",
        witness_nu_p=cfg["p"],
        witness_mc_samples=cfg["mc_samples"],
    )
    env = MaxSatEnv(
        [instance],
        d_max=instance.n_vars,
        horizon=max(12, cfg["rare_size"] + 2),
        seed=seed,
    )
    runner = RareDiscoveryRunner(
        env,
        config,
        advantage="mwrl_direct",
        optimizer="reinforce",
        seed=seed,
        antichain=antichain,
        budgets=tuple(cfg["budgets"]),
        direct_mc_samples=cfg["mc_samples"],
        direct_mc_p=cfg["p"],
        direct_mc_device=cfg["mc_device"],
        runner_seed=seed,
        rare_mask=rare_mask,
    )
    runner.train()
    return {
        "seed": seed,
        "records": runner.records,
        "final_found": sorted(runner.found),
        "rare_found": rare_mask in runner.found,
    }


def _deletion_runs(cfg: dict, instance: MaxSatInstance, antichain: list[int], rare_mask: int) -> list[dict]:
    full = (1 << instance.n_vars) - 1
    results: list[dict] = []
    for repeat in range(cfg["deletion_repeats"]):
        oracle = CountingOracle(instance)
        if not oracle(full):
            raise RuntimeError("full set is not sufficient")
        rng = np.random.default_rng([cfg["seed"], repeat])
        found: set[int] = set()
        records: list[dict] = []
        for budget in cfg["budgets"]:
            while oracle.requests + instance.n_vars <= budget:
                found.add(randomized_greedy_delete(oracle, full, rng))
            records.append(
                {
                    "budget": budget,
                    "queries": oracle.requests,
                    "recall": len(found) / len(antichain),
                    "rare_found": rare_mask in found,
                }
            )
        results.append({"repeat": repeat, "records": records})
    return results


def _aggregate(runs: list[dict], budgets: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    for budget in budgets:
        values = [record for run in runs for record in run["records"] if record["budget"] == budget]
        rows.append(
            {
                "budget": budget,
                "runs": len(values),
                "mean_recall": float(np.mean([value["recall"] for value in values])),
                "rare_recovery_rate": float(np.mean([value["rare_found"] for value in values])),
            }
        )
    return rows


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in raw.split(",")}))
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--singletons", type=int, default=7)
    parser.add_argument("--rare-size", type=int, default=7)
    parser.add_argument("--mwrl-seeds", type=_parse_int_tuple, default=(0, 1, 2, 3, 4))
    parser.add_argument("--deletion-repeats", type=int, default=500)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--groups-per-update", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=48)
    parser.add_argument("--mc-samples", type=int, default=20000)
    parser.add_argument(
        "--mc-device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        choices=["cpu", "mps", "cuda"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--budgets", type=_parse_int_tuple, default=DEFAULT_BUDGETS)
    parser.add_argument("--output", default="runs/maxsat_validations/rare_basin.json")
    args = parser.parse_args()
    if args.singletons < 1 or args.rare_size < 1:
        parser.error("witness blocks must be nonempty")

    instance, antichain, rare_mask = make_rare_basin_instance(args.singletons, args.rare_size)
    cfg = {
        "singletons": args.singletons,
        "rare_size": args.rare_size,
        "updates": args.updates,
        "groups_per_update": args.groups_per_update,
        "group_size": args.group_size,
        "mc_samples": args.mc_samples,
        "mc_device": args.mc_device,
        "p": args.p,
        "budgets": args.budgets,
        "deletion_repeats": args.deletion_repeats,
        "seed": args.seed,
    }
    deletion = _deletion_runs(cfg, instance, antichain, rare_mask)
    jobs = [(seed, cfg) for seed in args.mwrl_seeds]
    with Pool(min(args.workers, len(jobs))) as pool:
        mwrl = pool.map(_mwrl_worker, jobs)

    deletion_aggregate = _aggregate(deletion, args.budgets)
    mwrl_aggregate = _aggregate(mwrl, args.budgets)
    payload = {
        "config": {**cfg, "mwrl_seeds": args.mwrl_seeds},
        "geometry": {
            "n_vars": instance.n_vars,
            "antichain_size": len(antichain),
            "rare_deletion_probability": 1.0 / math.comb(instance.n_vars, args.singletons),
            "rare_mask": rare_mask,
        },
        "aggregate": {"deletion": deletion_aggregate, "mwrl_direct": mwrl_aggregate},
        "deletion_runs": deletion,
        "mwrl_runs": mwrl,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))

    print(
        f"rare deletion basin = 1/C({instance.n_vars},{args.singletons}) = "
        f"{payload['geometry']['rare_deletion_probability']:.7f}"
    )
    print(f"{'budget':>8} {'del recall':>11} {'del rare':>9} {'mwrl recall':>12} {'mwrl rare':>10}")
    for deletion_row, mwrl_row in zip(deletion_aggregate, mwrl_aggregate, strict=True):
        print(
            f"{deletion_row['budget']:8d} {deletion_row['mean_recall']:11.3f} "
            f"{deletion_row['rare_recovery_rate']:9.3f} {mwrl_row['mean_recall']:12.3f} "
            f"{mwrl_row['rare_recovery_rate']:10.3f}"
        )
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
