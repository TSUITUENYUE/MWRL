"""Ingest the measured-reaction CSV and turn it into per-task environments.

The CSV is the closed-world oracle: one row per real measured reaction. For each task
(substrate pair) we compute every reaction's deviation set (dimensions changed from the
baseline), mark successes at the threshold, and derive the exact ground-truth antichain --
the inclusion-minimal successful deviation sets. Unmeasured deviation sets are absent and are
treated as failures under the closed-world assumption.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mwrl_suzuki.fingerprint import reaction_fingerprint
from mwrl_suzuki.schema import ConditionSchema


def _is_subset(a: int, b: int) -> bool:
    """a is a subset of b (every bit of a is set in b)."""
    return (a | b) == b


def minimal_elements(masks: set[int]) -> list[int]:
    """Inclusion-minimal elements of a set of bitmasks (the antichain of generators)."""
    items = sorted(masks, key=lambda m: (bin(m).count("1"), m))
    minimal: list[int] = []
    for m in items:
        if not any(_is_subset(o, m) and o != m for o in items):
            minimal.append(m)
    return minimal


@dataclass
class ChemTask:
    task_id: str
    halide: str
    boron: str
    product: str | None
    n_dims: int
    successful_masks: frozenset[int]        # deviation sets of measured successes
    antichain: list[int]                    # inclusion-minimal successful deviation sets = M(c)
    yield_by_mask: dict[int, float]         # deviation set -> best measured yield
    max_yield: float
    threshold: float
    fingerprint: np.ndarray = field(default=None)  # type: ignore[assignment]
    dim_values: list[list[str]] = field(default_factory=list)   # candidate values per dimension (the verifier's search domain)
    baseline: list[str] = field(default_factory=list)           # baseline value per dimension
    yield_table: dict[tuple[str, ...], float] = field(default_factory=dict)  # full assignment -> best yield


def _threshold_for(yields: list[float], schema: ConditionSchema) -> float:
    if schema.threshold is not None:
        return float(schema.threshold)
    if schema.threshold_quantile is not None:
        return float(np.quantile(np.asarray(yields, dtype=float), schema.threshold_quantile))
    raise ValueError("schema defines neither an absolute threshold nor a quantile")


def load_dataset(
    csv_path: str | Path,
    schema: ConditionSchema,
    *,
    fingerprint_method: str = "drfp",
    n_bits: int = 2048,
    task_ids: Iterable[str] | None = None,
) -> list[ChemTask]:
    # ``task_ids`` keeps only those tasks' rows. Every derived quantity is per task and the
    # success threshold is absolute, so a subset load is identical to filtering a full load;
    # it just holds a fraction of the grid in memory (one worker needs only what it trains on).
    wanted = None if task_ids is None else set(task_ids)
    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            task_id = str(row[schema.task_col])
            if wanted is not None and task_id not in wanted:
                continue
            schema.validate_row(row)
            rows_by_task[task_id].append(row)

    dim_values = [list(schema.candidates[d]) for d in schema.dimensions]
    baseline = [schema.baseline[d] for d in schema.dimensions]

    tasks: list[ChemTask] = []
    for task_id, rows in rows_by_task.items():
        halides = {r[schema.halide_col] for r in rows}
        borons = {r[schema.boron_col] for r in rows}
        if len(halides) != 1 or len(borons) != 1:
            raise ValueError(f"task {task_id!r} has inconsistent substrate SMILES: {halides} x {borons}")
        halide, boron = halides.pop(), borons.pop()
        product = None
        if schema.product_col and schema.product_col in rows[0]:
            product = rows[0][schema.product_col] or None

        yields = [float(r[schema.yield_col]) for r in rows]
        threshold = _threshold_for(yields, schema)

        yield_by_mask: dict[int, float] = {}
        yield_table: dict[tuple[str, ...], float] = {}
        successful: set[int] = set()
        for row in rows:
            mask = schema.active_mask(row)
            y = float(row[schema.yield_col])
            yield_by_mask[mask] = max(yield_by_mask.get(mask, float("-inf")), y)
            key = tuple(str(row[d]) for d in schema.dimensions)
            yield_table[key] = max(yield_table.get(key, float("-inf")), y)
            if y >= threshold:
                successful.add(mask)

        tasks.append(
            ChemTask(
                task_id=task_id,
                halide=halide,
                boron=boron,
                product=product,
                n_dims=schema.n_dims,
                successful_masks=frozenset(successful),
                antichain=minimal_elements(successful),
                yield_by_mask=yield_by_mask,
                max_yield=max(yields),
                threshold=threshold,
                fingerprint=reaction_fingerprint(
                    halide, boron, product, n_bits=n_bits, method=fingerprint_method
                ),
                dim_values=dim_values,
                baseline=baseline,
                yield_table=yield_table,
            )
        )
    return tasks
