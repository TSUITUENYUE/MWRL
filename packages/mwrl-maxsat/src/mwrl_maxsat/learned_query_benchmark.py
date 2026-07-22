"""Archive training-time discoveries under a verifier-query budget.

The existing MaxSAT table evaluates a trained policy with a fresh rollout batch.  A
query-matched discovery comparison should also retain every minimal witness observed
during training.  This diagnostic records that archive for scalar MaxRL and for MWRL
using the direct shared-sample Monte Carlo estimator validated in
``credit_validation``.  The core runner and paper table are not modified.

Run one seed on the eight paper instances with

    python -m mwrl_maxsat.learned_query_benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from mwrl.config import MWRLConfig
from mwrl.runner import Runner

from mwrl_maxsat.credit_validation import direct_mc_credit_torch
from mwrl_maxsat.env import MaxSatEnv
from mwrl_maxsat.instance import generate_with_antichain

DEFAULT_BUDGETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 57600)


class DiscoveryRunner(Runner):
    """Runner that retains raw minimal terminals and optionally uses direct MC."""

    def __init__(
        self,
        *args,
        antichain: list[int],
        budgets: tuple[int, ...],
        direct_mc_samples: int,
        direct_mc_p: float,
        direct_mc_device: str,
        runner_seed: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.truth = set(antichain)
        self.budgets = budgets
        self.direct_mc_samples = direct_mc_samples
        self.direct_mc_p = direct_mc_p
        self.direct_mc_device = torch.device(direct_mc_device)
        self.direct_mc_generator = torch.Generator(device=direct_mc_device).manual_seed(runner_seed + 9_999_991)
        self.queries = 0
        self.found: set[int] = set()
        self.records: list[dict] = []

        n_vars = self.env.d_max
        self._states = np.arange(1 << n_vars, dtype=np.uint64)
        sizes = np.fromiter((int(state).bit_count() for state in self._states), dtype=np.int16)
        self._weights = np.power(direct_mc_p, sizes) * np.power(1.0 - direct_mc_p, n_vars - sizes)
        self._upsets = {mask: (self._states & np.uint64(mask)) == np.uint64(mask) for mask in antichain}
        full = np.zeros(self._states.shape[0], dtype=bool)
        for upset in self._upsets.values():
            full |= upset
        self._full_coverage = float(self._weights[full].sum())

    def _archive(self, rolls) -> None:
        for rollout in rolls:
            self.queries += 1
            if rollout.success and rollout.witness in self.truth:
                self.found.add(rollout.witness)
        while len(self.records) < len(self.budgets) and self.queries >= self.budgets[len(self.records)]:
            budget = self.budgets[len(self.records)]
            covered = np.zeros(self._states.shape[0], dtype=bool)
            for mask in self.found:
                covered |= self._upsets[mask]
            self.records.append(
                {
                    "budget": budget,
                    "queries": self.queries,
                    "found": len(self.found),
                    "antichain_size": len(self.truth),
                    "recall": len(self.found) / len(self.truth),
                    "normalized_upset_coverage": float(self._weights[covered].sum()) / self._full_coverage,
                }
            )

    def _rollout_group(self, context, k):
        rolls = super()._rollout_group(context, k)
        self._archive(rolls)
        return rolls

    def _group_advantages(self, group):
        if self.advantage != "mwrl_direct":
            return super()._group_advantages(group)
        _, centered = direct_mc_credit_torch(
            [rollout.witness for rollout in group],
            [rollout.success for rollout in group],
            n_vars=self.env.d_max,
            p=self.direct_mc_p,
            samples=self.direct_mc_samples,
            device=self.direct_mc_device,
            generator=self.direct_mc_generator,
        )
        credit = centered.to(self.device)
        scale = credit.abs().mean()
        return (credit / (scale + self.config.epsilon)).detach().cpu().tolist()


def _config(method: str, cfg: dict) -> MWRLConfig:
    common = dict(
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
    )
    if method == "maxrl":
        return MWRLConfig(**common)
    return MWRLConfig(
        witness_valuation="coverage",
        witness_center=True,
        witness_baseline="l2o",
        witness_normalize=True,
        witness_scale="mean",
        witness_nu="geometric",
        witness_nu_p=cfg["p"],
        witness_mc_samples=cfg["mc_samples"],
        **common,
    )


def _worker(job: tuple[str, int, int, dict]) -> dict:
    method, instance_id, seed, cfg = job
    torch.set_num_threads(1)
    instance, antichain = generate_with_antichain(
        cfg["n_vars"],
        cfg["n_clauses"],
        cfg["clause_len"],
        min_modes=cfg["min_modes"],
        max_modes=cfg["max_modes"],
        rng=np.random.default_rng(1000 + instance_id),
    )
    env = MaxSatEnv([instance], d_max=cfg["n_vars"], horizon=12, seed=seed)
    runner = DiscoveryRunner(
        env,
        _config(method, cfg),
        advantage="mwrl_direct" if method == "mwrl_direct" else "maxrl",
        optimizer="reinforce",
        seed=seed,
        antichain=antichain,
        budgets=tuple(cfg["budgets"]),
        direct_mc_samples=cfg["mc_samples"],
        direct_mc_p=cfg["p"],
        direct_mc_device=cfg["mc_device"],
        runner_seed=seed,
    )
    runner.train()
    return {
        "method": method,
        "instance": instance_id,
        "seed": seed,
        "antichain_size": len(antichain),
        "records": runner.records,
    }


def _aggregate(results: list[dict], methods: tuple[str, ...], budgets: tuple[int, ...]) -> dict:
    aggregate: dict[str, list[dict]] = {}
    for method in methods:
        rows: list[dict] = []
        for budget in budgets:
            values = [
                record
                for result in results
                if result["method"] == method
                for record in result["records"]
                if record["budget"] == budget
            ]
            if not values:
                continue
            recall = np.asarray([value["recall"] for value in values])
            coverage = np.asarray([value["normalized_upset_coverage"] for value in values])
            rows.append(
                {
                    "budget": budget,
                    "runs": len(values),
                    "mean_queries_used": float(np.mean([value["queries"] for value in values])),
                    "mean_recall": float(recall.mean()),
                    "std_recall": float(recall.std(ddof=1)) if len(recall) > 1 else 0.0,
                    "mean_normalized_upset_coverage": float(coverage.mean()),
                    "std_normalized_upset_coverage": (float(coverage.std(ddof=1)) if len(coverage) > 1 else 0.0),
                    "full_recovery_rate": float(np.mean(recall == 1.0)),
                }
            )
        aggregate[method] = rows
    return aggregate


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in raw.split(",")}))
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--seeds", type=_parse_int_tuple, default=(0,))
    parser.add_argument("--methods", default="mwrl_direct,maxrl")
    parser.add_argument("--n-vars", type=int, default=14)
    parser.add_argument("--n-clauses", type=int, default=10)
    parser.add_argument("--clause-len", type=int, default=3)
    parser.add_argument("--min-modes", type=int, default=6)
    parser.add_argument("--max-modes", type=int, default=20)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--groups-per-update", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=48)
    parser.add_argument("--mc-samples", type=int, default=20000)
    parser.add_argument(
        "--mc-device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        choices=["cpu", "mps", "cuda"],
    )
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="worker processes; zero chooses four for MPS jobs and all CPU cores otherwise",
    )
    parser.add_argument("--budgets", type=_parse_int_tuple, default=DEFAULT_BUDGETS)
    parser.add_argument("--output", default="runs/maxsat_validations/learned_query_archive.json")
    args = parser.parse_args()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = set(methods) - {"mwrl_direct", "maxrl"}
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")

    cfg = {
        "n_vars": args.n_vars,
        "n_clauses": args.n_clauses,
        "clause_len": args.clause_len,
        "min_modes": args.min_modes,
        "max_modes": args.max_modes,
        "updates": args.updates,
        "groups_per_update": args.groups_per_update,
        "group_size": args.group_size,
        "mc_samples": args.mc_samples,
        "mc_device": args.mc_device,
        "p": args.p,
        "budgets": args.budgets,
    }
    jobs = [
        (method, instance_id, seed, cfg)
        for method in methods
        for instance_id in range(args.instances)
        for seed in args.seeds
    ]
    if args.workers:
        processes = min(args.workers, len(jobs))
    elif args.mc_device == "mps" and "mwrl_direct" in methods:
        processes = min(4, len(jobs))
    else:
        processes = min(os.cpu_count() or 4, len(jobs))
    print(f"running {len(jobs)} jobs with {processes} workers", flush=True)
    with Pool(processes) as pool:
        results = pool.map(_worker, jobs)

    aggregate = _aggregate(results, methods, args.budgets)
    payload = {
        "config": {**cfg, "instances": args.instances, "seeds": args.seeds, "methods": methods},
        "aggregate": aggregate,
        "results": results,
        "notes": [
            "all raw minimal terminals encountered during training are retained",
            "one terminal sufficiency evaluation counts as one verifier query",
            "mwrl_direct uses the appendix shared-sample estimator without changing core training code",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))

    for method in methods:
        print(method)
        for row in aggregate[method]:
            print(
                f"  {row['budget']:6d} calls  recall={row['mean_recall']:.3f}  "
                f"coverage={row['mean_normalized_upset_coverage']:.3f}"
            )
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
