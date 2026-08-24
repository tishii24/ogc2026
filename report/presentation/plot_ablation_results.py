#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

VERSIONS = [
    ("122", "Solution"),
    ("000-no-preoptimize", "w/o Preoptimization"),
    ("000-no-reconstruct", "w/o Large Reconstruction"),
    ("000-no-rolling-horizon", "w/o Rolling Horizon"),
]

VERSION_COLORS = ["#4389DC", "#D78329", "#789180", "#716F9C"]
COMPONENT_COLORS = ["#4389DC", "#D78329", "#A15D99"]
TEXT_COLOR = "#3F3F3F"
GRID_COLOR = "#AEB4BB"


def run_stats(root: Path, *extra_args: str) -> Any:
    command = [
        sys.executable,
        str(root / "tools/stats.py"),
        "--suite",
        "suites/final.json",
        "--tl",
        "300",
        "--json",
        *extra_args,
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )


def compute_relative_scores(
    matrix: dict[str, Any],
) -> tuple[list[str], dict[str, np.ndarray]]:
    case_labels = matrix["headers"][5:]
    target_versions = {version for version, _ in VERSIONS}
    rows_by_version = {
        row[0]: row for row in matrix["rows"] if row[0] in target_versions
    }
    objectives = {
        version: np.array([float(value) for value in rows_by_version[version][5:]])
        for version, _ in VERSIONS
    }
    best = np.min(np.stack(list(objectives.values())), axis=0)
    relative_scores = {
        version: np.divide(best, values, out=np.ones_like(best), where=values != 0)
        for version, values in objectives.items()
    }
    return case_labels, relative_scores


def plot_relative_series(
    axis: Any,
    x: np.ndarray,
    values_by_version: dict[str, np.ndarray],
    *,
    markersize: float,
) -> None:
    for index, (version, label) in enumerate(VERSIONS):
        axis.plot(
            x,
            values_by_version[version],
            label=label,
            color=VERSION_COLORS[index],
            linewidth=3.2 if index == 0 else 2.0,
            marker="o",
            markersize=markersize if index == 0 else markersize - 1.0,
            markeredgewidth=0,
            alpha=1.0,
            zorder=4 if index == 0 else 3,
        )
    axis.axhline(1.0, color=TEXT_COLOR, linewidth=1.2, linestyle=(0, (5, 5)), alpha=0.55)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_case_relative_score(
    case_labels: list[str], relative_scores: dict[str, np.ndarray], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(13.333, 7.5), dpi=240)
    x = np.arange(1, len(case_labels) + 1)
    plot_relative_series(axis, x, relative_scores, markersize=4.5)

    all_values = np.concatenate(list(relative_scores.values()))
    lower = max(0.0, np.floor((all_values.min() - 0.03) * 20) / 20)
    axis.set_xlim(0.5, len(case_labels) + 0.5)
    axis.set_ylim(lower, 1.015)
    axis.set_xlabel("Case", labelpad=12, fontsize=17)
    axis.set_ylabel("Relative Score", labelpad=12, fontsize=17)
    axis.set_xticks(x)
    axis.set_xticklabels(case_labels, fontsize=10)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=3.0,
        columnspacing=2.0,
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.86)
    figure.savefig(output, dpi=240, transparent=True)
    plt.close(figure)


def save_grouped_relative_score(
    root: Path,
    case_labels: list[str],
    relative_scores: dict[str, np.ndarray],
    output: Path,
) -> None:
    block_counts = []
    bay_counts = []
    for case_label in case_labels:
        with (root / "in/train-final" / f"prob_{case_label}.json").open(encoding="utf-8") as file:
            problem = json.load(file)
        block_counts.append(len(problem["blocks"]))
        bay_counts.append(len(problem["bays"]))

    grouped_series = []
    for group_values in (np.array(block_counts), np.array(bay_counts)):
        categories = np.array(sorted(set(group_values)))
        means = {
            version: np.array(
                [relative_scores[version][group_values == category].mean() for category in categories]
            )
            for version, _ in VERSIONS
        }
        grouped_series.append((categories, means))

    figure, axes = plt.subplots(2, 1, figsize=(13.333, 7.5), dpi=240, sharey=True)
    all_means = np.concatenate(
        [values for _, means in grouped_series for values in means.values()]
    )
    lower = max(0.0, np.floor((all_means.min() - 0.02) * 20) / 20)

    for axis, (categories, means), xlabel in zip(
        axes,
        grouped_series,
        ("Number of Blocks", "Number of Bays"),
    ):
        x = np.arange(len(categories))
        plot_relative_series(axis, x, means, markersize=7.0)
        axis.set_xlim(-0.2, len(categories) - 0.8)
        axis.set_ylim(lower, 1.015)
        axis.set_xticks(x)
        axis.set_xticklabels(categories)
        axis.set_xlabel(xlabel, labelpad=8, fontsize=16)

    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=3.0,
        columnspacing=2.0,
    )
    figure.supylabel("Mean Relative Score", x=0.035, fontsize=17)
    figure.subplots_adjust(left=0.1, right=0.985, bottom=0.1, top=0.84, hspace=0.42)
    figure.savefig(output, dpi=240, transparent=True)
    plt.close(figure)


def save_objective_breakdown(summaries: list[dict[str, Any]], output: Path) -> None:
    summary_by_version = {summary["version"]: summary for summary in summaries}
    labels = [
        "Solution",
        "w/o\nPreoptimization",
        "w/o Large\nReconstruction",
        "w/o Rolling\nHorizon",
    ]
    z1 = np.array(
        [summary_by_version[version]["total_obj1"] for version, _ in VERSIONS],
        dtype=float,
    ) / 1e6
    z2 = np.array(
        [summary_by_version[version]["total_obj2"] for version, _ in VERSIONS],
        dtype=float,
    ) / 1e6
    z3 = np.array(
        [summary_by_version[version]["total_obj3"] for version, _ in VERSIONS],
        dtype=float,
    ) / 1e6
    totals = z1 + z2 + z3

    figure, axis = plt.subplots(figsize=(13.333, 7.5), dpi=240)
    x = np.arange(len(VERSIONS))
    width = 0.62

    axis.bar(
        x,
        z1,
        width,
        label="Total Tardiness",
        color=COMPONENT_COLORS[0],
        edgecolor=TEXT_COLOR,
        linewidth=0.8,
    )
    axis.bar(
        x,
        z2,
        width,
        bottom=z1,
        label="Workload Imbalance",
        color=COMPONENT_COLORS[1],
        edgecolor=TEXT_COLOR,
        linewidth=0.8,
    )
    axis.bar(
        x,
        z3,
        width,
        bottom=z1 + z2,
        label="Total Preference Score",
        color=COMPONENT_COLORS[2],
        edgecolor=TEXT_COLOR,
        linewidth=0.8,
    )

    for index, total in enumerate(totals):
        axis.text(
            index,
            total + max(totals) * 0.018,
            f"{total:.1f}M",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
        axis.text(
            index,
            z1[index] * 0.5,
            f"{z1[index]:.1f}M",
            ha="center",
            va="center",
            fontsize=12,
            color=TEXT_COLOR,
        )
        axis.text(
            index,
            z1[index] + z2[index] + z3[index] * 0.5,
            f"{z3[index]:.1f}M",
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT_COLOR,
        )

    z2_note = "Workload imbalance: " + " / ".join(f"{value:.2f}M" for value in z2)
    axis.text(
        0.5,
        -0.17,
        z2_note + "  (Solution / Preoptimization / Large Reconstruction / Rolling Horizon)",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color=COMPONENT_COLORS[1],
    )

    axis.set_ylabel("Objective Value (millions)", labelpad=12, fontsize=17)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=13)
    axis.set_ylim(0, max(totals) * 1.13)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        columnspacing=1.8,
    )
    figure.subplots_adjust(left=0.1, right=0.985, bottom=0.23, top=0.86)
    figure.savefig(output, dpi=240, transparent=True)
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_plot()
    summaries = run_stats(root)
    matrix = run_stats(root, "--matrix", "--cell-mode", "absolute")
    case_labels, relative_scores = compute_relative_scores(matrix)

    case_output = output_dir / "case-relative-score.png"
    grouped_output = output_dir / "grouped-relative-score.png"
    breakdown_output = output_dir / "objective-breakdown.png"
    save_case_relative_score(case_labels, relative_scores, case_output)
    save_grouped_relative_score(root, case_labels, relative_scores, grouped_output)
    save_objective_breakdown(summaries, breakdown_output)
    print(case_output)
    print(grouped_output)
    print(breakdown_output)


if __name__ == "__main__":
    main()
