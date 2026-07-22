"""Validate MaxSAT Monte Carlo coverage credit against exact lattice enumeration.

This module does not modify training.  It compares the currently deployed union-
difference estimator with a direct shared-sample estimator of the same leave-two-out
credit.  Exact targets are computed by enumerating all ``2**n`` lattice states, which
is feasible for the paper's ``n=14`` instances and avoids using inclusion-exclusion as
its own ground truth.

Run the local audit with

    python -m mwrl_maxsat.credit_validation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mwrl.credit import coverage_l2o_advantages, nu_upset_geometric

from mwrl_maxsat.instance import generate_with_antichain


def _containment_matrix(states: np.ndarray, masks: list[int], successes: list[bool]) -> np.ndarray:
    cover = np.zeros((states.shape[0], len(masks)), dtype=bool)
    for index, (mask, success) in enumerate(zip(masks, successes, strict=True)):
        if success:
            value = np.uint64(mask)
            cover[:, index] = (states & value) == value
    return cover


def _credit_from_cover(cover: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return raw marginal credit and leave-two-out centered credit.

    For a sampled lattice state ``T``, a proposal has raw marginal support exactly
    when it is the unique proposal contained by ``T``.  The leave-two-out baseline
    for proposal ``i`` is ``1/(K-1)`` whenever exactly one other proposal is
    contained.  Averaging these indicators gives an O(NK) shared-sample estimator.
    """
    count = cover.sum(axis=1)
    if weights is None:
        raw = (cover & (count[:, None] == 1)).mean(axis=0, dtype=np.float64)
        if cover.shape[1] > 1:
            baseline = ((count[:, None] - cover) == 1).mean(axis=0, dtype=np.float64)
            baseline /= cover.shape[1] - 1
        else:
            baseline = np.zeros(cover.shape[1], dtype=np.float64)
    else:
        raw = (weights[:, None] * (cover & (count[:, None] == 1))).sum(axis=0)
        if cover.shape[1] > 1:
            baseline = (weights[:, None] * ((count[:, None] - cover) == 1)).sum(axis=0)
            baseline /= cover.shape[1] - 1
        else:
            baseline = np.zeros(cover.shape[1], dtype=np.float64)
    return raw, raw - baseline


def exact_credit(masks: list[int], successes: list[bool], *, n_vars: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    states = np.arange(1 << n_vars, dtype=np.uint64)
    sizes = np.fromiter((int(state).bit_count() for state in states), dtype=np.int16)
    weights = np.power(p, sizes) * np.power(1.0 - p, n_vars - sizes)
    cover = _containment_matrix(states, masks, successes)
    return _credit_from_cover(cover, weights)


def direct_mc_credit(
    masks: list[int],
    successes: list[bool],
    *,
    n_vars: int,
    p: float,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    draws = rng.random((samples, n_vars)) < p
    states = np.zeros(samples, dtype=np.uint64)
    for variable in range(n_vars):
        states |= draws[:, variable].astype(np.uint64) << np.uint64(variable)
    cover = _containment_matrix(states, masks, successes)
    return _credit_from_cover(cover)


def direct_mc_credit_torch(
    masks: list[int],
    successes: list[bool],
    *,
    n_vars: int,
    p: float,
    samples: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel shared-sample credit on CPU, CUDA, or Apple MPS."""
    target = torch.device(device)
    required = torch.tensor(
        [[(mask >> variable) & 1 for variable in range(n_vars)] for mask in masks],
        dtype=torch.float32,
        device=target,
    )
    success_tensor = torch.tensor(successes, dtype=torch.bool, device=target)
    required_sizes = required.sum(dim=1)
    draws = (torch.rand((samples, n_vars), device=target, generator=generator) < p).to(torch.float32)
    cover = (draws @ required.T == required_sizes[None, :]) & success_tensor[None, :]
    count = cover.sum(dim=1)
    raw = (cover & (count[:, None] == 1)).to(torch.float32).mean(dim=0)
    if len(masks) > 1:
        baseline = (count[:, None] - cover.to(torch.int64) == 1).to(torch.float32).mean(dim=0) / (len(masks) - 1)
    else:
        baseline = torch.zeros_like(raw)
    return raw, raw - baseline


def _synchronize(device: torch.device | str) -> None:
    target = torch.device(device)
    if target.type == "mps":
        torch.mps.synchronize()
    elif target.type == "cuda":
        torch.cuda.synchronize(target)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _sign_accuracy(estimate: np.ndarray, target: np.ndarray, tolerance: float = 1e-12) -> float:
    def signs(values: np.ndarray) -> np.ndarray:
        return np.where(values > tolerance, 1, np.where(values < -tolerance, -1, 0))

    return float(np.mean(signs(estimate) == signs(target)))


def _support_scores(estimate: np.ndarray, target: np.ndarray, tolerance: float = 1e-12) -> dict:
    predicted = estimate > tolerance
    actual = target > tolerance
    true_positive = int(np.logical_and(predicted, actual).sum())
    precision = true_positive / max(int(predicted.sum()), 1)
    recall = true_positive / max(int(actual.sum()), 1)
    return {
        "support_precision": precision,
        "support_recall": recall,
        "support_accuracy": float(np.mean(predicted == actual)),
    }


def _metrics(
    estimate: np.ndarray,
    target: np.ndarray,
    *,
    elapsed: float,
) -> dict:
    error = estimate - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "relative_l1": float(np.abs(error).sum() / max(np.abs(target).sum(), 1e-15)),
        "cosine": _cosine(estimate, target),
        "sign_accuracy": _sign_accuracy(estimate, target),
        "elapsed_seconds": elapsed,
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    result: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[f"mean_{key}"] = float(values.mean())
        result[f"std_{key}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def _make_clean_group(*, n_vars: int, modes: int, seed: int) -> tuple[list[int], list[bool], int]:
    instance, antichain = generate_with_antichain(
        n_vars,
        10,
        3,
        min_modes=modes,
        max_modes=20,
        rng=np.random.default_rng(seed),
        tries=2000,
    )
    del instance
    return list(antichain[:modes]), [True] * modes, len(antichain)


def _make_mixed_group(minimal_masks: list[int], *, n_vars: int, seed: int) -> tuple[list[int], list[bool]]:
    """Mix unique modes, duplicate modes, dominated supersets, and failures."""
    rng = np.random.default_rng(seed)
    masks = list(minimal_masks)
    successes = [True] * len(masks)

    duplicated = minimal_masks[: len(minimal_masks) // 2]
    masks.extend(duplicated)
    successes.extend([True] * len(duplicated))

    for mask in duplicated:
        absent = [v for v in range(n_vars) if not (mask >> v) & 1]
        if absent:
            masks.append(mask | (1 << int(rng.choice(absent))))
            successes.append(True)

    failure_count = len(minimal_masks)
    masks.extend([0] * failure_count)
    successes.extend([False] * failure_count)
    order = rng.permutation(len(masks))
    return [masks[int(i)] for i in order], [successes[int(i)] for i in order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-vars", type=int, default=14)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--samples", default="256,1024,5000,20000")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--current-repetitions", type=int, default=3)
    parser.add_argument(
        "--direct-device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
        choices=["cpu", "mps", "cuda"],
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--instance-seed", type=int, default=1000)
    parser.add_argument("--output", default="runs/maxsat_validations/mc_credit.json")
    args = parser.parse_args()
    sample_counts = tuple(sorted({int(value) for value in args.samples.split(",")}))
    if not sample_counts or sample_counts[0] < 1:
        parser.error("--samples must contain positive integers")
    if args.modes <= 12:
        parser.error("--modes must exceed 12 to exercise the current Monte Carlo branch")

    masks, successes, available_modes = _make_clean_group(n_vars=args.n_vars, modes=args.modes, seed=args.instance_seed)
    exact_raw, exact_centered = exact_credit(masks, successes, n_vars=args.n_vars, p=args.p)
    mixed_masks, mixed_successes = _make_mixed_group(masks, n_vars=args.n_vars, seed=args.seed)
    mixed_exact_raw, mixed_exact_centered = exact_credit(mixed_masks, mixed_successes, n_vars=args.n_vars, p=args.p)

    rows: list[dict] = []
    nu = nu_upset_geometric(args.p)
    for samples in sample_counts:
        direct_clean: list[dict] = []
        direct_mixed: list[dict] = []
        direct_support: list[dict] = []
        warmup_generator = torch.Generator(device=args.direct_device).manual_seed(args.seed + 10_000_019 * samples - 1)
        direct_mc_credit_torch(
            masks,
            successes,
            n_vars=args.n_vars,
            p=args.p,
            samples=samples,
            device=args.direct_device,
            generator=warmup_generator,
        )
        direct_mc_credit_torch(
            mixed_masks,
            mixed_successes,
            n_vars=args.n_vars,
            p=args.p,
            samples=samples,
            device=args.direct_device,
            generator=warmup_generator,
        )
        _synchronize(args.direct_device)
        for repetition in range(args.repetitions):
            generator = torch.Generator(device=args.direct_device).manual_seed(
                args.seed + 10_000_019 * samples + 101 * repetition
            )
            _synchronize(args.direct_device)
            started = time.perf_counter()
            _, centered_tensor = direct_mc_credit_torch(
                masks,
                successes,
                n_vars=args.n_vars,
                p=args.p,
                samples=samples,
                device=args.direct_device,
                generator=generator,
            )
            _synchronize(args.direct_device)
            centered = centered_tensor.detach().cpu().numpy()
            direct_clean.append(_metrics(centered, exact_centered, elapsed=time.perf_counter() - started))

            generator = torch.Generator(device=args.direct_device).manual_seed(
                args.seed + 10_000_019 * samples + 101 * repetition + 1
            )
            _synchronize(args.direct_device)
            started = time.perf_counter()
            mixed_raw_tensor, mixed_centered_tensor = direct_mc_credit_torch(
                mixed_masks,
                mixed_successes,
                n_vars=args.n_vars,
                p=args.p,
                samples=samples,
                device=args.direct_device,
                generator=generator,
            )
            _synchronize(args.direct_device)
            mixed_raw = mixed_raw_tensor.detach().cpu().numpy()
            mixed_centered = mixed_centered_tensor.detach().cpu().numpy()
            direct_mixed.append(
                _metrics(
                    mixed_centered,
                    mixed_exact_centered,
                    elapsed=time.perf_counter() - started,
                )
            )
            direct_support.append(_support_scores(mixed_raw, mixed_exact_raw))

        current: list[dict] = []
        for _ in range(args.current_repetitions):
            started = time.perf_counter()
            estimate = np.asarray(
                coverage_l2o_advantages(
                    masks,
                    successes,
                    nu_upset=nu,
                    mc_p=args.p,
                    mc_samples=samples,
                )
            )
            current.append(_metrics(estimate, exact_centered, elapsed=time.perf_counter() - started))

        rows.append(
            {
                "samples": samples,
                "direct_clean": _aggregate(direct_clean),
                "direct_mixed": _aggregate(direct_mixed),
                "direct_mixed_support": _aggregate(direct_support),
                "current_clean": _aggregate(current),
            }
        )

    payload = {
        "config": {
            "n_vars": args.n_vars,
            "requested_modes": args.modes,
            "available_modes": available_modes,
            "p": args.p,
            "samples": sample_counts,
            "direct_repetitions": args.repetitions,
            "current_repetitions": args.current_repetitions,
            "direct_device": args.direct_device,
            "seed": args.seed,
            "instance_seed": args.instance_seed,
        },
        "groups": {
            "clean_k": len(masks),
            "mixed_k": len(mixed_masks),
            "mixed_successes": int(sum(mixed_successes)),
            "mixed_exact_positive_raw_support": int((mixed_exact_raw > 1e-12).sum()),
        },
        "exact": {
            "clean_raw": exact_raw.tolist(),
            "clean_centered": exact_centered.tolist(),
            "mixed_raw": mixed_exact_raw.tolist(),
            "mixed_centered": mixed_exact_centered.tolist(),
        },
        "results": rows,
        "implementation_notes": [
            "current_clean calls the deployed coverage_l2o_advantages implementation",
            "the deployed implementation creates an unseeded RNG internally",
            "direct_* uses one shared lattice-sample batch and O(NK) containment statistics",
            "direct_* parallelizes containment and reduction with Torch on the configured device",
            "Monte Carlo activation depends on pruned successful antichain size, not K",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))

    print(
        f"clean K={len(masks)}, mixed K={len(mixed_masks)}, "
        f"mixed positive support={payload['groups']['mixed_exact_positive_raw_support']}"
    )
    print(f"{'N':>8} {'current cos':>12} {'direct cos':>11} {'speedup':>10} {'support':>10}")
    for row in rows:
        current = row["current_clean"]
        direct = row["direct_clean"]
        speedup = current["mean_elapsed_seconds"] / max(direct["mean_elapsed_seconds"], 1e-12)
        support = row["direct_mixed_support"]
        print(
            f"{row['samples']:8d} {current['mean_cosine']:12.3f} "
            f"{direct['mean_cosine']:11.3f} {speedup:10.1f} "
            f"{support['mean_support_accuracy']:10.3f}"
        )
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
