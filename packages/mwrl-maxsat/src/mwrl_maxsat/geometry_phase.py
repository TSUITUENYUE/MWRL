"""Controlled antichain-geometry phase diagram for Minimal-Witness RL.

This experiment isolates the geometry of an exact monotone predicate

    s_A(S) = 1 iff some M in A is a subset of S,

whose minimal-witness antichain is exactly the planted family ``A``.  It varies
antichain size, witness-size imbalance, structural overlap, and L/K while holding
the policy architecture, optimizer steps, and terminal-verifier calls fixed.

The runner batches all trajectories on CPU and computes all grouped MWRL credits in
one shared-sample MPS operation.  The primary baselines are learned methods with the
same terminal-query budget: size-aware scalar policy gradient and an up-set GFlowNet.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from mwrl.modules import Actor
from torch import nn

OVERLAP_QUANTILES = {"low": 0.1, "mid": 0.5, "high": 0.9}


@dataclass(frozen=True)
class PlantedPredicate:
    n_vars: int
    masks: tuple[int, ...]
    antichain_size: int
    size_gap: int
    overlap_level: str
    target_overlap: float
    achieved_overlap: float
    overlap_min: float
    overlap_max: float
    mean_jaccard: float
    mean_measure_overlap: float
    mean_size: float
    size_cv: float
    exclusive_mass_min: float
    exclusive_mass_mean: float
    exclusive_mass_cv: float
    coverage_value: float
    predicate_seed: int


def _mask_from_indices(indices) -> int:
    return sum(1 << int(index) for index in indices)


def _sizes_for(antichain_size: int, size_gap: int) -> list[int]:
    if size_gap not in {0, 2, 4}:
        raise ValueError("size_gap must be one of 0, 2, or 4")
    if size_gap == 0:
        return [4] * antichain_size
    if antichain_size % 2:
        raise ValueError("imbalanced profiles require an even antichain size")
    low = 4 - size_gap // 2
    high = 4 + size_gap // 2
    return [low] * (antichain_size // 2) + [high] * (antichain_size // 2)


def _is_antichain(masks: list[int]) -> bool:
    return len(set(masks)) == len(masks) and not any(
        left != right and (left & right) in {left, right} for i, left in enumerate(masks) for right in masks[i + 1 :]
    )


def _random_family(n_vars: int, sizes: list[int], rng: np.random.Generator) -> list[int]:
    for _ in range(200):
        family: list[int] = []
        order = rng.permutation(len(sizes))
        failed = False
        for index in order:
            size = sizes[int(index)]
            for _ in range(2000):
                candidate = _mask_from_indices(rng.choice(n_vars, size=size, replace=False))
                if candidate in family:
                    continue
                if any((candidate & other) in {candidate, other} for other in family):
                    continue
                family.append(candidate)
                break
            else:
                failed = True
                break
        if not failed and len(family) == len(sizes):
            return family
    raise RuntimeError("could not generate a valid planted antichain")


def _overlap_coefficient(left: int, right: int) -> float:
    return (left & right).bit_count() / min(left.bit_count(), right.bit_count())


def _mean_overlap(masks: list[int]) -> float:
    if len(masks) < 2:
        return 0.0
    return float(
        np.mean([_overlap_coefficient(left, right) for i, left in enumerate(masks) for right in masks[i + 1 :]])
    )


def _replacement_score(masks: list[int], index: int, candidate: int, current: float) -> float:
    count = len(masks) * (len(masks) - 1) / 2
    if count == 0:
        return 0.0
    old_sum = sum(_overlap_coefficient(masks[index], other) for j, other in enumerate(masks) if j != index)
    new_sum = sum(_overlap_coefficient(candidate, other) for j, other in enumerate(masks) if j != index)
    return current + (new_sum - old_sum) / count


def _valid_replacement(masks: list[int], index: int, candidate: int) -> bool:
    for j, other in enumerate(masks):
        if j == index:
            continue
        if candidate == other or (candidate & other) in {candidate, other}:
            return False
    return True


def _optimize_overlap(
    n_vars: int,
    sizes: list[int],
    *,
    rng: np.random.Generator,
    objective: str,
    target: float | None = None,
    steps: int = 4000,
    restarts: int = 3,
) -> tuple[list[int], float]:
    best_family: list[int] | None = None
    best_score: float | None = None
    best_loss = math.inf
    for _ in range(restarts):
        family = _random_family(n_vars, sizes, rng)
        score = _mean_overlap(family)
        for step in range(steps):
            index = int(rng.integers(len(family)))
            candidate = _mask_from_indices(rng.choice(n_vars, size=family[index].bit_count(), replace=False))
            if not _valid_replacement(family, index, candidate):
                continue
            proposal = _replacement_score(family, index, candidate, score)
            if objective == "min":
                old_loss, new_loss = score, proposal
            elif objective == "max":
                old_loss, new_loss = -score, -proposal
            elif objective == "target" and target is not None:
                old_loss, new_loss = abs(score - target), abs(proposal - target)
            else:
                raise ValueError(f"unsupported overlap objective {objective!r}")
            temperature = 0.01 * (1.0 - step / max(steps, 1)) + 1e-4
            if new_loss <= old_loss or rng.random() < math.exp((old_loss - new_loss) / temperature):
                family[index] = candidate
                score = proposal

            loss = score if objective == "min" else -score if objective == "max" else abs(score - target)
            if loss < best_loss:
                best_loss = loss
                best_family = list(family)
                best_score = score
    if best_family is None or best_score is None:
        raise RuntimeError("overlap optimization failed")
    return best_family, float(best_score)


def _family_statistics(masks: list[int], n_vars: int, p: float) -> dict[str, float]:
    sizes = np.asarray([mask.bit_count() for mask in masks], dtype=float)
    pair_jaccard = [
        (left & right).bit_count() / (left | right).bit_count()
        for i, left in enumerate(masks)
        for right in masks[i + 1 :]
    ]
    measure_overlap = [
        p ** ((left | right).bit_count())
        / (p ** left.bit_count() + p ** right.bit_count() - p ** ((left | right).bit_count()))
        for i, left in enumerate(masks)
        for right in masks[i + 1 :]
    ]

    rng_seed = sum((index + 1) * mask for index, mask in enumerate(masks)) % (2**63 - 1)
    rng = np.random.default_rng(rng_seed)
    measure_samples = 100_000
    draws = rng.random((measure_samples, n_vars)) < p
    states = np.zeros(measure_samples, dtype=np.uint64)
    for variable in range(n_vars):
        states |= draws[:, variable].astype(np.uint64) << np.uint64(variable)
    cover = np.column_stack([(states & np.uint64(mask)) == np.uint64(mask) for mask in masks])
    cover_count = cover.sum(axis=1)
    exclusive = (cover & (cover_count[:, None] == 1)).mean(axis=0)
    coverage_value = float((cover_count > 0).mean())
    return {
        "mean_jaccard": float(np.mean(pair_jaccard)) if pair_jaccard else 0.0,
        "mean_measure_overlap": float(np.mean(measure_overlap)) if measure_overlap else 0.0,
        "mean_size": float(sizes.mean()),
        "size_cv": float(sizes.std() / sizes.mean()),
        "exclusive_mass_min": float(exclusive.min()),
        "exclusive_mass_mean": float(exclusive.mean()),
        "exclusive_mass_cv": float(exclusive.std() / max(exclusive.mean(), 1e-15)),
        "coverage_value": coverage_value,
    }


def generate_predicate(
    *,
    n_vars: int,
    antichain_size: int,
    size_gap: int,
    overlap_level: str,
    predicate_seed: int,
    p: float,
    optimization_steps: int,
) -> PlantedPredicate:
    if overlap_level not in OVERLAP_QUANTILES:
        raise ValueError(f"unknown overlap level {overlap_level!r}")
    sizes = _sizes_for(antichain_size, size_gap)
    base_seed = 1_000_003 * predicate_seed + 10_007 * antichain_size + 101 * size_gap + 17 * n_vars
    minimum_family, minimum = _optimize_overlap(
        n_vars,
        sizes,
        rng=np.random.default_rng(base_seed + 1),
        objective="min",
        steps=optimization_steps,
    )
    maximum_family, maximum = _optimize_overlap(
        n_vars,
        sizes,
        rng=np.random.default_rng(base_seed + 2),
        objective="max",
        steps=optimization_steps,
    )
    del minimum_family, maximum_family
    target = minimum + OVERLAP_QUANTILES[overlap_level] * (maximum - minimum)
    masks, achieved = _optimize_overlap(
        n_vars,
        sizes,
        rng=np.random.default_rng(base_seed + 3 + 100 * list(OVERLAP_QUANTILES).index(overlap_level)),
        objective="target",
        target=target,
        steps=optimization_steps,
    )
    if not _is_antichain(masks):
        raise AssertionError("planted family is not an antichain")
    permutation = np.random.default_rng(base_seed + 99).permutation(n_vars)
    permuted = tuple(
        sorted(
            _mask_from_indices(permutation[index] for index in range(n_vars) if (mask >> index) & 1) for mask in masks
        )
    )
    stats = _family_statistics(list(permuted), n_vars, p)
    return PlantedPredicate(
        n_vars=n_vars,
        masks=permuted,
        antichain_size=antichain_size,
        size_gap=size_gap,
        overlap_level=overlap_level,
        target_overlap=target,
        achieved_overlap=achieved,
        overlap_min=minimum,
        overlap_max=maximum,
        predicate_seed=predicate_seed,
        **stats,
    )


def _mask_membership(masks: tuple[int, ...] | list[int], n_vars: int) -> torch.Tensor:
    return torch.tensor(
        [[(mask >> variable) & 1 for variable in range(n_vars)] for mask in masks],
        dtype=torch.float32,
    )


@dataclass
class BatchRollout:
    terminal: torch.Tensor
    success: torch.Tensor
    minimal: torch.Tensor
    logp_sum: torch.Tensor
    entropy_mean: torch.Tensor
    sizes: torch.Tensor


def _sample_batch(
    actor: Actor,
    *,
    truth: torch.Tensor,
    batch_size: int,
    horizon: int,
    generator: torch.Generator,
    with_grad: bool,
) -> BatchRollout:
    del generator  # Categorical sampling follows the seeded global Torch RNG.
    n_vars = truth.shape[1]
    terminal = torch.zeros((batch_size, n_vars), dtype=torch.bool)
    steps = torch.zeros(batch_size, dtype=torch.int64)
    active = torch.ones(batch_size, dtype=torch.bool)
    logp_sum = torch.zeros(batch_size, dtype=torch.float32)
    entropy_sum = torch.zeros((), dtype=torch.float32)
    action_count = 0

    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        while bool(active.any()):
            indices = torch.nonzero(active, as_tuple=False).squeeze(1)
            membership = terminal[indices]
            obs = torch.cat(
                [membership.to(torch.float32), (steps[indices] / horizon).to(torch.float32)[:, None]],
                dim=1,
            )
            action_mask = torch.cat([~membership, torch.ones((len(indices), 1), dtype=torch.bool)], dim=1)
            distribution = actor.distribution(obs, action_mask)
            actions = distribution.sample()
            logp_sum = logp_sum.index_add(0, indices, distribution.log_prob(actions))
            entropy_sum = entropy_sum + distribution.entropy().sum()
            action_count += len(indices)

            stopped = actions == n_vars
            adding_indices = indices[~stopped]
            adding_actions = actions[~stopped]
            if len(adding_indices):
                terminal[adding_indices, adding_actions] = True
                steps[adding_indices] += 1
            active[indices[stopped]] = False
            active[adding_indices[steps[adding_indices] >= horizon]] = False

    terminal_float = terminal.to(torch.float32)
    truth_sizes = truth.sum(dim=1)
    covers = terminal_float @ truth.T == truth_sizes[None, :]
    success = covers.any(dim=1)
    minimal = (terminal[:, None, :] == truth.to(torch.bool)[None, :, :]).all(dim=2).any(dim=1)
    return BatchRollout(
        terminal=terminal,
        success=success,
        minimal=minimal,
        logp_sum=logp_sum,
        entropy_mean=entropy_sum / max(action_count, 1),
        sizes=terminal.sum(dim=1).to(torch.float32),
    )


class VectorLearner:
    def __init__(
        self,
        predicate: PlantedPredicate,
        *,
        method: str,
        group_size: int,
        trajectories_per_update: int,
        updates: int,
        horizon: int,
        learning_rate: float,
        entropy_coef: float,
        size_penalty: float,
        p: float,
        mc_samples: int,
        mc_device: str,
        seed: int,
        normalization: str = "group",
    ) -> None:
        if trajectories_per_update % group_size:
            raise ValueError("trajectories_per_update must be divisible by group_size")
        if method not in {"mwrl", "scalar_size", "gflownet"}:
            raise ValueError(f"unsupported method {method!r}")
        if normalization not in {"none", "group", "update"}:
            raise ValueError(f"unsupported normalization {normalization!r}")
        torch.manual_seed(seed)
        self.predicate = predicate
        self.method = method
        self.group_size = group_size
        self.batch_size = trajectories_per_update
        self.groups = trajectories_per_update // group_size
        self.updates = updates
        self.horizon = horizon
        self.entropy_coef = entropy_coef
        self.size_penalty = size_penalty
        self.p = p
        self.mc_samples = mc_samples
        self.mc_device = torch.device(mc_device)
        self.normalization = normalization
        self.generator = torch.Generator(device="cpu").manual_seed(seed + 31)
        self.mc_generator = torch.Generator(device=mc_device).manual_seed(seed + 104_729)
        self.truth = _mask_membership(predicate.masks, predicate.n_vars)
        self.actor = Actor(predicate.n_vars + 1, predicate.n_vars + 1, hidden_dim=128)
        self.log_z = nn.Parameter(torch.zeros(())) if method == "gflownet" else None
        parameters = list(self.actor.parameters())
        if self.log_z is not None:
            parameters.append(self.log_z)
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        self.found: set[int] = set()
        self.history: list[dict] = []
        self._powers = torch.tensor([1 << variable for variable in range(predicate.n_vars)])

    def _archive(self, rollout: BatchRollout) -> None:
        integer_masks = (rollout.terminal.to(torch.int64) * self._powers).sum(dim=1)
        truth = set(self.predicate.masks)
        self.found.update(int(mask) for mask in integer_masks[rollout.minimal] if int(mask) in truth)

    def _mwrl_advantage(self, rollout: BatchRollout) -> torch.Tensor:
        required = rollout.terminal.to(device=self.mc_device, dtype=torch.float32)
        success = rollout.success.to(device=self.mc_device)
        draws = (
            torch.rand(
                (self.mc_samples, self.predicate.n_vars),
                device=self.mc_device,
                generator=self.mc_generator,
            )
            < self.p
        ).to(torch.float32)
        required_sizes = required.sum(dim=1)
        cover = (draws @ required.T == required_sizes[None, :]) & success[None, :]
        cover = cover.reshape(self.mc_samples, self.groups, self.group_size)
        count = cover.sum(dim=2)
        raw = (cover & (count[:, :, None] == 1)).to(torch.float32).mean(dim=0)
        if self.group_size > 1:
            baseline = (count[:, :, None] - cover.to(torch.int64) == 1).to(torch.float32).mean(dim=0) / (
                self.group_size - 1
            )
        else:
            baseline = torch.zeros_like(raw)
        centered = raw - baseline
        if self.normalization == "none":
            advantage = centered
        elif self.normalization == "group":
            scale = centered.abs().mean(dim=1, keepdim=True)
            advantage = centered / (scale + 1e-6)
        else:
            scale = centered.abs().mean()
            advantage = centered / (scale + 1e-6)
        return advantage.reshape(-1).to("cpu")

    def _scalar_advantage(self, rollout: BatchRollout) -> torch.Tensor:
        reward = rollout.success.to(torch.float32) - self.size_penalty * rollout.sizes
        reward = reward.reshape(self.groups, self.group_size)
        centered = reward - reward.mean(dim=1, keepdim=True)
        scale = reward.std(dim=1, keepdim=True, unbiased=False)
        return (centered / (scale + 1e-6)).reshape(-1)

    def train(self) -> dict:
        started = time.perf_counter()
        for update in range(self.updates):
            rollout = _sample_batch(
                self.actor,
                truth=self.truth,
                batch_size=self.batch_size,
                horizon=self.horizon,
                generator=self.generator,
                with_grad=True,
            )
            self._archive(rollout)
            if self.method == "gflownet":
                assert self.log_z is not None
                log_reward = torch.where(
                    rollout.success,
                    rollout.sizes * math.log(self.p),
                    torch.full_like(rollout.sizes, math.log(1e-4)),
                )
                log_backward = -torch.lgamma(rollout.sizes + 1.0)
                residual = self.log_z + rollout.logp_sum - log_reward - log_backward
                loss = residual.square().mean()
            else:
                advantage = self._mwrl_advantage(rollout) if self.method == "mwrl" else self._scalar_advantage(rollout)
                loss = -(advantage.detach() * rollout.logp_sum).mean()
                loss = loss - self.entropy_coef * rollout.entropy_mean
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.optimizer.step()
            self.history.append(
                {
                    "update": update + 1,
                    "queries": (update + 1) * self.batch_size,
                    "archive_recall": len(self.found) / self.predicate.antichain_size,
                    "success_rate": float(rollout.success.to(torch.float32).mean()),
                    "born_minimal_rate": float(rollout.minimal.to(torch.float32).mean()),
                }
            )
        return {"train_seconds": time.perf_counter() - started}

    def evaluate(self, samples: int) -> dict:
        rollout = _sample_batch(
            self.actor,
            truth=self.truth,
            batch_size=samples,
            horizon=self.horizon,
            generator=self.generator,
            with_grad=False,
        )
        integer_masks = (rollout.terminal.to(torch.int64) * self._powers).sum(dim=1)
        found = set(int(mask) for mask in integer_masks[rollout.minimal]) & set(self.predicate.masks)
        small_size = min(mask.bit_count() for mask in self.predicate.masks)
        large_size = max(mask.bit_count() for mask in self.predicate.masks)
        small_truth = {mask for mask in self.predicate.masks if mask.bit_count() == small_size}
        large_truth = {mask for mask in self.predicate.masks if mask.bit_count() == large_size}
        return {
            "eval_recall": len(found) / self.predicate.antichain_size,
            "eval_found": len(found),
            "eval_success_rate": float(rollout.success.to(torch.float32).mean()),
            "eval_born_minimal_rate": float(rollout.minimal.to(torch.float32).mean()),
            "small_witness_recall": len(found & small_truth) / len(small_truth),
            "large_witness_recall": len(found & large_truth) / len(large_truth),
        }


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in raw.split(",") if value)


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in raw.split(",") if value)


def _parse_strings(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _preset(args) -> dict:
    if args.preset == "pilot":
        return {
            "antichain_sizes": (8, 32),
            "size_gaps": (0, 4),
            "overlaps": ("low", "high"),
            "ratios": (0.5, 2.0),
            "predicate_seeds": (0, 1),
            "policy_seeds": (0,),
        }
    return {
        "antichain_sizes": (4, 8, 16, 32),
        "size_gaps": (0, 2, 4),
        "overlaps": ("low", "mid", "high"),
        "ratios": (0.5, 1.0, 2.0),
        "predicate_seeds": tuple(range(12)),
        "policy_seeds": (0, 1, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--n-vars", type=int, default=20)
    parser.add_argument("--antichain-sizes", type=_parse_ints)
    parser.add_argument("--size-gaps", type=_parse_ints)
    parser.add_argument("--overlaps", type=_parse_strings)
    parser.add_argument("--ratios", type=_parse_floats)
    parser.add_argument("--predicate-seeds", type=_parse_ints)
    parser.add_argument("--policy-seeds", type=_parse_ints)
    parser.add_argument("--methods", type=_parse_strings, default=("mwrl", "scalar_size", "gflownet"))
    parser.add_argument("--trajectories-per-update", type=int, default=256)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--size-penalty", type=float, default=0.05)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--mc-samples", type=int, default=5000)
    parser.add_argument(
        "--mc-device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        choices=["cpu", "mps", "cuda"],
    )
    parser.add_argument("--optimization-steps", type=int, default=2500)
    parser.add_argument("--output", default="runs/maxsat_geometry/phase_pilot.json")
    args = parser.parse_args()
    levels = _preset(args)
    antichain_sizes = args.antichain_sizes or levels["antichain_sizes"]
    size_gaps = args.size_gaps or levels["size_gaps"]
    overlaps = args.overlaps or levels["overlaps"]
    ratios = args.ratios or levels["ratios"]
    predicate_seeds = args.predicate_seeds or levels["predicate_seeds"]
    policy_seeds = args.policy_seeds or levels["policy_seeds"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predicates: dict[tuple, PlantedPredicate] = {}
    results: list[dict] = []
    for antichain_size in antichain_sizes:
        for size_gap in size_gaps:
            for overlap in overlaps:
                for predicate_seed in predicate_seeds:
                    key = (antichain_size, size_gap, overlap, predicate_seed)
                    predicates[key] = generate_predicate(
                        n_vars=args.n_vars,
                        antichain_size=antichain_size,
                        size_gap=size_gap,
                        overlap_level=overlap,
                        predicate_seed=predicate_seed,
                        p=args.p,
                        optimization_steps=args.optimization_steps,
                    )

    jobs = sum(
        1
        for predicate in predicates.values()
        for ratio in ratios
        for method in args.methods
        for _ in policy_seeds
        if round(predicate.antichain_size / ratio) > 1
    )
    print(f"generated {len(predicates)} predicates; running {jobs} learned jobs", flush=True)
    completed = 0
    for predicate in predicates.values():
        for ratio in ratios:
            group_size = int(round(predicate.antichain_size / ratio))
            if group_size <= 1 or args.trajectories_per_update % group_size:
                continue
            for method in args.methods:
                for policy_seed in policy_seeds:
                    learner = VectorLearner(
                        predicate,
                        method=method,
                        group_size=group_size,
                        trajectories_per_update=args.trajectories_per_update,
                        updates=args.updates,
                        horizon=args.horizon,
                        learning_rate=args.learning_rate,
                        entropy_coef=args.entropy,
                        size_penalty=args.size_penalty,
                        p=args.p,
                        mc_samples=args.mc_samples,
                        mc_device=args.mc_device,
                        seed=policy_seed,
                    )
                    timing = learner.train()
                    evaluation = learner.evaluate(args.eval_samples)
                    result = {
                        "method": method,
                        "policy_seed": policy_seed,
                        "group_size": group_size,
                        "l_over_k": predicate.antichain_size / group_size,
                        "train_queries": args.updates * args.trajectories_per_update,
                        "eval_queries": args.eval_samples,
                        "archive_recall": len(learner.found) / predicate.antichain_size,
                        "history": learner.history,
                        **timing,
                        **evaluation,
                        "predicate": asdict(predicate),
                    }
                    results.append(result)
                    completed += 1
                    payload = {
                        "config": {
                            **vars(args),
                            "antichain_sizes": antichain_sizes,
                            "size_gaps": size_gaps,
                            "overlaps": overlaps,
                            "ratios": ratios,
                            "predicate_seeds": predicate_seeds,
                            "policy_seeds": policy_seeds,
                        },
                        "results": results,
                    }
                    output.write_text(json.dumps(payload, indent=2))
                    print(
                        f"[{completed:4d}/{jobs}] L={predicate.antichain_size:2d} "
                        f"gap={predicate.size_gap} overlap={predicate.overlap_level:<4} "
                        f"L/K={predicate.antichain_size / group_size:g} {method:<11} "
                        f"archive={result['archive_recall']:.3f} eval={result['eval_recall']:.3f} "
                        f"{timing['train_seconds']:.2f}s",
                        flush=True,
                    )
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
