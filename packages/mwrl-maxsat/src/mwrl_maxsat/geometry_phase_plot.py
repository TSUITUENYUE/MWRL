"""Render the antichain-geometry phase diagram from ``geometry_phase`` results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


def _cell_key(result: dict) -> tuple:
    predicate = result["predicate"]
    return (
        predicate["antichain_size"],
        predicate["size_gap"],
        predicate["overlap_level"],
        float(result["l_over_k"]),
    )


def _paired_rows(results: list[dict], metric: str) -> list[dict]:
    grouped: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        grouped[_cell_key(result)][result["method"]].append(result)

    rows: list[dict] = []
    for cell, methods in grouped.items():
        if "mwrl" not in methods:
            continue
        baseline_candidates = [name for name in ("scalar_size", "gflownet") if name in methods]
        baseline = max(
            baseline_candidates,
            key=lambda name: np.mean([result[metric] for result in methods[name]]),
        )
        by_run: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for method in ("mwrl", baseline):
            for result in methods[method]:
                predicate = result["predicate"]
                run_key = (predicate["predicate_seed"], result["policy_seed"])
                by_run[run_key][method] = result
        paired = [pair for pair in by_run.values() if len(pair) == 2]
        if not paired:
            continue
        gains = np.asarray([pair["mwrl"][metric] - pair[baseline][metric] for pair in paired])
        mwrl = np.asarray([pair["mwrl"][metric] for pair in paired])
        base = np.asarray([pair[baseline][metric] for pair in paired])
        representative = paired[0]["mwrl"]["predicate"]

        predicate_gains: dict[int, list[float]] = defaultdict(list)
        for pair in paired:
            seed = pair["mwrl"]["predicate"]["predicate_seed"]
            predicate_gains[int(seed)].append(pair["mwrl"][metric] - pair[baseline][metric])
        clusters = np.asarray([np.mean(values) for values in predicate_gains.values()])
        if len(clusters) >= 2:
            rng = np.random.default_rng(20260711)
            bootstrap = np.asarray([rng.choice(clusters, size=len(clusters), replace=True).mean() for _ in range(5000)])
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        else:
            ci_low = ci_high = float("nan")
        baseline_mean = float(base.mean())
        rows.append(
            {
                "antichain_size": cell[0],
                "size_gap": cell[1],
                "overlap_level": cell[2],
                "l_over_k": cell[3],
                "baseline": baseline,
                "train_queries": paired[0]["mwrl"]["train_queries"],
                "eval_queries": paired[0]["mwrl"]["eval_queries"],
                "n_runs": len(paired),
                "n_predicates": len(predicate_gains),
                "mwrl_recall": float(mwrl.mean()),
                "baseline_recall": baseline_mean,
                "gain": float(gains.mean()),
                "gain_ci_low": float(ci_low),
                "gain_ci_high": float(ci_high),
                "vi_gap_closed": float(gains.mean() / max(1.0 - baseline_mean, 1e-12)),
                "achieved_overlap": float(np.mean([pair["mwrl"]["predicate"]["achieved_overlap"] for pair in paired])),
                "mean_jaccard": float(np.mean([pair["mwrl"]["predicate"]["mean_jaccard"] for pair in paired])),
                "exclusive_mass_cv": float(
                    np.mean([pair["mwrl"]["predicate"]["exclusive_mass_cv"] for pair in paired])
                ),
                "mean_size": representative["mean_size"],
            }
        )
    return rows


def _contrast_color(value: float, norm, cmap) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.5 else "#202020"


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.18,
        1.12,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def render(rows: list[dict], output: Path, metric: str) -> None:
    antichain_sizes = sorted({row["antichain_size"] for row in rows})
    size_gaps = sorted({row["size_gap"] for row in rows})
    overlap_levels = [level for level in ("low", "mid", "high") if level in {row["overlap_level"] for row in rows}]
    ratios = sorted({row["l_over_k"] for row in rows})
    lookup = {(row["antichain_size"], row["size_gap"], row["overlap_level"], row["l_over_k"]): row for row in rows}

    max_abs = max(0.15, max(abs(row["gain"]) for row in rows))
    max_abs = math.ceil(max_abs * 20) / 20
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = plt.get_cmap("RdBu_r")

    figure = plt.figure(figsize=(7.15, 2.0 * len(size_gaps) + 2.25))
    grid = figure.add_gridspec(
        len(size_gaps) + 1,
        len(antichain_sizes),
        height_ratios=[1.0] * len(size_gaps) + [1.15],
        hspace=0.62,
        wspace=0.30,
        left=0.08,
        right=0.91,
        bottom=0.10,
        top=0.92,
    )
    image = None
    panel_index = 0
    for row_index, size_gap in enumerate(size_gaps):
        for column_index, antichain_size in enumerate(antichain_sizes):
            ax = figure.add_subplot(grid[row_index, column_index])
            matrix = np.full((len(overlap_levels), len(ratios)), np.nan)
            cell_rows: dict[tuple[int, int], dict] = {}
            for y, overlap in enumerate(overlap_levels):
                for x, ratio in enumerate(ratios):
                    row = lookup.get((antichain_size, size_gap, overlap, ratio))
                    if row is not None:
                        matrix[y, x] = row["gain"]
                        cell_rows[(y, x)] = row
            image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for (y, x), row in cell_rows.items():
                significant = (
                    row["n_predicates"] >= 5
                    and np.isfinite(row["gain_ci_low"])
                    and (row["gain_ci_low"] > 0 or row["gain_ci_high"] < 0)
                )
                label = f"{row['gain']:+.2f}{'•' if significant else ''}"
                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color=_contrast_color(row["gain"], norm, cmap),
                    fontweight="bold" if significant else "normal",
                )
            ax.set_xticks(range(len(ratios)))
            ax.set_xticklabels([f"{ratio:g}" for ratio in ratios])
            ax.set_yticks(range(len(overlap_levels)))
            ax.set_yticklabels(overlap_levels if column_index == 0 else [])
            if row_index == len(size_gaps) - 1:
                ax.set_xlabel("proposal scarcity  $L/K$")
            if column_index == 0:
                ax.set_ylabel(f"overlap\n$\\Delta_s={size_gap}$")
            if row_index == 0:
                ax.set_title(f"$L={antichain_size}$", fontsize=8.2, pad=5)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            _panel_label(ax, chr(ord("a") + panel_index))
            panel_index += 1

    if image is not None:
        colorbar_ax = figure.add_axes([0.93, 0.47, 0.015, 0.34])
        colorbar = figure.colorbar(image, cax=colorbar_ax)
        colorbar.set_label("MWRL recall gain", fontsize=7)
        colorbar.ax.tick_params(labelsize=6.5, length=2)

    marginal_ax = figure.add_subplot(grid[-1, : max(1, len(antichain_sizes) // 2)])
    colors = {size: color for size, color in zip(antichain_sizes, ["#7884B4", "#B64342", "#42949E", "#9A4D8E"])}
    for antichain_size in antichain_sizes:
        for size_gap, linestyle in zip(size_gaps, ["-", "--", ":"]):
            means = []
            lows = []
            highs = []
            for ratio in ratios:
                selected = [
                    row
                    for row in rows
                    if row["antichain_size"] == antichain_size
                    and row["size_gap"] == size_gap
                    and row["l_over_k"] == ratio
                ]
                means.append(np.mean([row["gain"] for row in selected]))
                lows.append(np.mean([row["gain_ci_low"] for row in selected]))
                highs.append(np.mean([row["gain_ci_high"] for row in selected]))
            marginal_ax.plot(
                ratios,
                means,
                color=colors[antichain_size],
                linestyle=linestyle,
                marker="o",
                ms=3.5,
                lw=1.3,
                label=f"$L={antichain_size}$, $\\Delta_s={size_gap}$",
            )
            if np.all(np.isfinite(lows)) and np.all(np.isfinite(highs)):
                marginal_ax.fill_between(ratios, lows, highs, color=colors[antichain_size], alpha=0.10)
    marginal_ax.axhline(0, color="#767676", lw=0.8, linestyle="--")
    marginal_ax.set_xscale("log", base=2)
    marginal_ax.set_xticks(ratios)
    marginal_ax.set_xticklabels([f"{ratio:g}" for ratio in ratios])
    marginal_ax.set_xlabel("proposal scarcity  $L/K$")
    marginal_ax.set_ylabel("mean recall gain")
    marginal_ax.legend(fontsize=5.8, ncol=2, loc="best")
    _panel_label(marginal_ax, chr(ord("a") + panel_index))
    panel_index += 1

    scatter_ax = figure.add_subplot(grid[-1, max(1, len(antichain_sizes) // 2) :])
    markers = {gap: marker for gap, marker in zip(size_gaps, ["o", "s", "^"])}
    for row in rows:
        scatter_ax.scatter(
            row["achieved_overlap"],
            row["vi_gap_closed"],
            s=20 + 1.4 * row["antichain_size"],
            marker=markers[row["size_gap"]],
            facecolor=colors[row["antichain_size"]],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
        )
    scatter_ax.axhline(0, color="#767676", lw=0.8, linestyle="--")
    scatter_ax.axhline(1, color="#A8A8A8", lw=0.8, linestyle=":")
    scatter_ax.set_xlabel("achieved structural overlap")
    scatter_ax.set_ylabel("fraction of MW-VI gap closed")
    scatter_ax.text(
        0.98,
        0.04,
        "MW-VI recall = 1\nmarker: size imbalance\npoint area: antichain size",
        transform=scatter_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#606060",
    )
    _panel_label(scatter_ax, chr(ord("a") + panel_index))

    figure.suptitle(
        "Relational credit becomes decisive for large balanced witness families",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
    )
    if metric == "archive_recall":
        subtitle = f"Cumulative born-minimal recall after {rows[0]['train_queries']:,} matched verifier calls"
    else:
        subtitle = f"Fresh-policy recall from {rows[0]['eval_queries']:,} matched post-training rollouts"
    figure.text(0.08, 0.955, subtitle, ha="left", va="top", fontsize=6.8, color="#606060")
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        figure.savefig(
            output.with_suffix(f".{extension}"),
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    summary_path = output.with_name(f"{output.name}_source.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/maxsat_geometry/phase_pilot.json")
    parser.add_argument("--output", default="runs/maxsat_geometry/phase_pilot")
    parser.add_argument("--metric", choices=["archive_recall", "eval_recall"], default="archive_recall")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    rows = _paired_rows(payload["results"], args.metric)
    if not rows:
        raise RuntimeError("no paired MWRL/baseline cells found")
    render(rows, Path(args.output), args.metric)
    print(f"rendered {args.output}.svg/.pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
