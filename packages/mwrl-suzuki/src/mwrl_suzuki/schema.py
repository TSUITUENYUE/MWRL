"""Infer and validate the fixed Suzuki condition schema from the measurement CSV.

The generated CSV is the complete data boundary between data generation and the method. Its
``active_set`` and ``active_mask`` columns fix the dimension-to-bit mapping, its zero-mask rows
fix the shared baseline, its observed values fix the candidate sets, and
``success_threshold`` fixes the verifier threshold. No generator configuration is imported.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ConditionSchema:
    dimensions: tuple[str, ...]              # ordered; index in this tuple = bit position
    candidates: dict[str, tuple[str, ...]]   # allowed values per dimension
    baseline: dict[str, str]                 # the default value per dimension (shared across tasks)
    task_col: str = "task_id"
    halide_col: str = "halide_smiles"
    boron_col: str = "boron_smiles"
    product_col: str | None = "product_smiles"
    yield_col: str = "yield"
    threshold: float | None = None           # absolute success cutoff on yield
    threshold_quantile: float | None = None  # per-task quantile cutoff (used only when threshold is None)

    @property
    def n_dims(self) -> int:
        return len(self.dimensions)

    def active_mask(self, row: dict[str, Any]) -> int:
        """Bitmask of dimensions whose value in ``row`` differs from the baseline."""
        mask = 0
        for i, dim in enumerate(self.dimensions):
            if str(row[dim]) != str(self.baseline[dim]):
                mask |= 1 << i
        return mask

    def validate_row(self, row: dict[str, Any]) -> None:
        for dim in self.dimensions:
            if dim not in row:
                raise ValueError(f"row is missing condition column {dim!r}")
            value = str(row[dim])
            if value not in self.candidates[dim]:
                raise ValueError(
                    f"value {value!r} for dimension {dim!r} is not in its candidate set "
                    f"{self.candidates[dim]}"
                )
        for col in (self.task_col, self.halide_col, self.boron_col, self.yield_col):
            if col not in row:
                raise ValueError(f"row is missing required column {col!r}")

    @classmethod
    def from_csv(cls, path: str | Path) -> ConditionSchema:
        """Infer the shared condition schema and verify the CSV metadata contract.

        The contract deliberately uses only ordinary CSV columns:

        * ``active_set`` is a JSON list of dimensions changed from the baseline.
        * ``active_mask`` is the corresponding integer bitmask.
        * ``success_threshold`` is one global finite yield threshold.
        * every task has exactly one row with ``active_mask == 0``.
        * structured singleton coverage exposes every candidate value for every task.

        Requiring singleton coverage makes the bit order identifiable without a sidecar file.
        It is already part of the generated Suzuki benchmark design.
        """

        csv_path = Path(path)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            rows = list(reader)

        if not fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        if not rows:
            raise ValueError(f"CSV has no measurement rows: {csv_path}")

        required = {
            "task_id",
            "halide_smiles",
            "boron_smiles",
            "yield",
            "active_set",
            "active_mask",
            "success_threshold",
        }
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                "CSV is not a self-contained Suzuki dataset; missing required columns "
                f"{missing}"
            )

        parsed: list[tuple[int, dict[str, str], tuple[str, ...], int, float]] = []
        bit_by_name: dict[str, int] = {}
        name_by_bit: dict[int, str] = {}
        dimension_names: set[str] = set()
        threshold: float | None = None

        for line_number, row in enumerate(rows, start=2):
            task_id = str(row["task_id"]).strip()
            if not task_id:
                raise ValueError(f"line {line_number}: task_id is empty")

            active_set = _parse_active_set(row["active_set"], line_number)
            dimension_names.update(active_set)
            try:
                active_mask = int(str(row["active_mask"]).strip())
            except ValueError as exc:
                raise ValueError(
                    f"line {line_number}: active_mask must be an integer"
                ) from exc
            if active_mask < 0:
                raise ValueError(f"line {line_number}: active_mask must be nonnegative")
            if active_mask.bit_count() != len(active_set):
                raise ValueError(
                    f"line {line_number}: active_mask has {active_mask.bit_count()} bits "
                    f"but active_set has {len(active_set)} dimensions"
                )

            try:
                row_threshold = float(row["success_threshold"])
            except ValueError as exc:
                raise ValueError(
                    f"line {line_number}: success_threshold must be numeric"
                ) from exc
            if not math.isfinite(row_threshold):
                raise ValueError(f"line {line_number}: success_threshold must be finite")
            if threshold is None:
                threshold = row_threshold
            elif not math.isclose(row_threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"line {line_number}: success_threshold={row_threshold} disagrees "
                    f"with the dataset threshold {threshold}"
                )

            if len(active_set) == 1:
                if active_mask == 0 or active_mask & (active_mask - 1):
                    raise ValueError(
                        f"line {line_number}: singleton active_set requires a one-bit active_mask"
                    )
                name = active_set[0]
                bit = active_mask.bit_length() - 1
                if name in bit_by_name and bit_by_name[name] != bit:
                    raise ValueError(
                        f"line {line_number}: dimension {name!r} is assigned to multiple bits"
                    )
                if bit in name_by_bit and name_by_bit[bit] != name:
                    raise ValueError(
                        f"line {line_number}: bit {bit} is assigned to multiple dimensions"
                    )
                bit_by_name[name] = bit
                name_by_bit[bit] = name

            parsed.append((line_number, row, active_set, active_mask, row_threshold))

        if not dimension_names:
            raise ValueError("CSV exposes no active condition dimensions")
        unresolved = sorted(dimension_names - set(bit_by_name))
        if unresolved:
            raise ValueError(
                "cannot infer bit positions because these dimensions have no singleton rows: "
                f"{unresolved}"
            )
        expected_bits = set(range(len(dimension_names)))
        if set(name_by_bit) != expected_bits:
            raise ValueError(
                "active_mask bit positions must be contiguous from zero; "
                f"observed {sorted(name_by_bit)}"
            )
        dimensions = tuple(name_by_bit[index] for index in range(len(dimension_names)))
        missing_dimension_columns = [name for name in dimensions if name not in fieldnames]
        if missing_dimension_columns:
            raise ValueError(
                f"active_set names missing as CSV columns: {missing_dimension_columns}"
            )

        baseline_rows: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
        for line_number, row, active_set, active_mask, _ in parsed:
            expected_mask = sum(1 << bit_by_name[name] for name in active_set)
            if active_mask != expected_mask:
                raise ValueError(
                    f"line {line_number}: active_mask={active_mask} does not encode "
                    f"active_set={list(active_set)} under the inferred bit order"
                )
            if active_mask == 0:
                if active_set:
                    raise ValueError(
                        f"line {line_number}: zero active_mask requires an empty active_set"
                    )
                baseline_rows[str(row["task_id"])].append((line_number, row))

        task_ids = {str(row["task_id"]) for _, row, _, _, _ in parsed}
        for task_id in sorted(task_ids):
            count = len(baseline_rows[task_id])
            if count != 1:
                raise ValueError(
                    f"task {task_id!r} has {count} baseline rows; exactly one is required"
                )

        first_task = next(iter(sorted(task_ids)))
        _, first_baseline_row = baseline_rows[first_task][0]
        baseline = {name: str(first_baseline_row[name]) for name in dimensions}
        for task_id in sorted(task_ids):
            line_number, row = baseline_rows[task_id][0]
            actual = {name: str(row[name]) for name in dimensions}
            if actual != baseline:
                raise ValueError(
                    f"line {line_number}: task {task_id!r} uses a different baseline; "
                    "all tasks must share one condition baseline"
                )

        candidates_seen: dict[str, list[str]] = {
            name: [baseline[name]] for name in dimensions
        }
        task_candidates: dict[str, dict[str, set[str]]] = {
            task_id: {name: set() for name in dimensions} for task_id in task_ids
        }
        for line_number, row, active_set, active_mask, _ in parsed:
            task_id = str(row["task_id"])
            changed = tuple(
                name for name in dimensions if str(row[name]) != baseline[name]
            )
            if changed != active_set:
                raise ValueError(
                    f"line {line_number}: active_set={list(active_set)} disagrees with "
                    f"the dimensions changed from baseline={list(changed)}"
                )
            expected_mask = sum(1 << index for index, name in enumerate(dimensions) if name in changed)
            if active_mask != expected_mask:
                raise ValueError(
                    f"line {line_number}: active_mask={active_mask} disagrees with changed columns"
                )
            for name in dimensions:
                value = str(row[name])
                task_candidates[task_id][name].add(value)
                if value not in candidates_seen[name]:
                    candidates_seen[name].append(value)

        candidates = {
            name: tuple(values) for name, values in candidates_seen.items()
        }
        for task_id in sorted(task_ids):
            for name in dimensions:
                missing_values = set(candidates[name]) - task_candidates[task_id][name]
                if missing_values:
                    raise ValueError(
                        f"task {task_id!r} does not expose the shared candidate values "
                        f"{sorted(missing_values)} for dimension {name!r}"
                    )

        assert threshold is not None
        return cls(
            dimensions=dimensions,
            candidates=candidates,
            baseline=baseline,
            product_col="product_smiles" if "product_smiles" in fieldnames else None,
            threshold=threshold,
            threshold_quantile=None,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConditionSchema:
        cfg = yaml.safe_load(Path(path).read_text())["schema"]
        dims = cfg["dimensions"]
        names = tuple(d["name"] for d in dims)
        candidates = {d["name"]: tuple(str(v) for v in d["values"]) for d in dims}
        baseline = {d["name"]: str(d["baseline"]) for d in dims}
        for name in names:
            if baseline[name] not in candidates[name]:
                raise ValueError(f"baseline {baseline[name]!r} for {name!r} is not in its candidate set")
        cols = cfg.get("columns", {})
        succ = cfg.get("success", {})
        return cls(
            dimensions=names,
            candidates=candidates,
            baseline=baseline,
            task_col=cols.get("task", "task_id"),
            halide_col=cols.get("halide", "halide_smiles"),
            boron_col=cols.get("boron", "boron_smiles"),
            product_col=cols.get("product", "product_smiles"),
            yield_col=cols.get("yield", "yield"),
            threshold=succ.get("threshold"),
            threshold_quantile=succ.get("quantile"),
        )


def _parse_active_set(value: str, line_number: int) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"line {line_number}: active_set must be a JSON list of dimension names"
        ) from exc
    if not isinstance(parsed, list) or any(not isinstance(name, str) for name in parsed):
        raise ValueError(
            f"line {line_number}: active_set must be a JSON list of dimension names"
        )
    names = tuple(parsed)
    if len(names) != len(set(names)):
        raise ValueError(f"line {line_number}: active_set contains duplicate dimensions")
    return names
