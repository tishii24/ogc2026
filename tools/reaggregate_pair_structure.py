#!/usr/bin/env python3
"""Reaggregate pair-structure metrics without rerunning pair placement search."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

EPS = 1e-9


@dataclass
class Metrics:
    n_good: int = 0
    w_large: float = 0.0
    w_overlap: float = 0.0


@dataclass
class Record:
    row: dict[str, str]
    current: Metrics
    best_n_good: float
    best_w_large: float
    best_w_overlap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reaggregate pair-structure metrics at another good ratio."
    )
    parser.add_argument("--good-ratio", type=float, default=1.20)
    parser.add_argument(
        "--details", default="log/pair-structure-details.csv"
    )
    parser.add_argument(
        "--summary", default="log/pair-structure-summary.csv"
    )
    parser.add_argument(
        "--output", default="log/pair-structure-summary-ratio-1.20.csv"
    )
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def source_key(version: str, timelimit: str) -> tuple[str, float]:
    return version, float(timelimit)


def projection_area(
    root: Path,
    case: str,
    block_id: int,
    orient_idx: int,
    problems: dict[str, dict],
    areas: dict[tuple[str, int, int], float],
) -> float:
    key = (case, block_id, orient_idx)
    if key in areas:
        return areas[key]
    if case not in problems:
        import json

        with resolve(root, case).open(encoding="utf-8") as f:
            problems[case] = json.load(f)
    layers = problems[case]["blocks"][block_id]["shape"][orient_idx]["layers"]
    area = float(unary_union([Polygon(layer) for layer in layers]).area)
    areas[key] = area
    return area


def bay_area(root: Path, case: str, bay_id: int, problems: dict[str, dict]) -> float:
    if case not in problems:
        import json

        with resolve(root, case).open(encoding="utf-8") as f:
            problems[case] = json.load(f)
    bay = problems[case]["bays"][bay_id]
    return float(bay["width"] * bay["height"])


def load_metrics(
    root: Path, details_path: Path, good_ratio: float
) -> dict[tuple[str, str, tuple[str, float]], Metrics]:
    metrics: dict[tuple[str, str, tuple[str, float]], Metrics] = defaultdict(Metrics)
    problems: dict[str, dict] = {}
    areas: dict[tuple[str, int, int], float] = {}

    with details_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["actual_compactness"]) > good_ratio * float(
                row["best_compactness"]
            ) + EPS:
                continue

            case = row["case"]
            key = (case, row["source_kind"], source_key(row["version"], row["timelimit"]))
            metric = metrics[key]
            metric.n_good += 1

            stored_large = float(row["large_weight"])
            if stored_large > 0.0:
                large = stored_large
                overlap = float(row["overlap_weight"])
            else:
                block_i = int(row["block_i"])
                block_j = int(row["block_j"])
                orient_i = int(row["orient_i"])
                orient_j = int(row["orient_j"])
                area_i = projection_area(
                    root, case, block_i, orient_i, problems, areas
                )
                area_j = projection_area(
                    root, case, block_j, orient_j, problems, areas
                )
                area_bay = bay_area(root, case, int(row["bay_id"]), problems)
                large = min(area_i, area_j) / area_bay
                overlap = max(
                    0.0,
                    float(row["area_sum"]) - float(row["actual_hull_area"]),
                ) / area_bay

            metric.w_large += large
            metric.w_overlap += overlap

    return metrics


def load_records(
    summary_path: Path,
    metrics: dict[tuple[str, str, tuple[str, float]], Metrics],
) -> list[Record]:
    records = []
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case = row["case"]
            current_key = source_key(row["current_version"], row["current_timelimit"])
            current = metrics[(case, "current", current_key)]
            best_sources = []
            for source in row["best_sources"].split("|"):
                version, timelimit = source.rsplit("/", 1)
                best_sources.append(metrics[(case, "best", source_key(version, timelimit))])
            records.append(
                Record(
                    row=row,
                    current=current,
                    best_n_good=statistics.median(
                        metric.n_good for metric in best_sources
                    ),
                    best_w_large=statistics.median(
                        metric.w_large for metric in best_sources
                    ),
                    best_w_overlap=statistics.median(
                        metric.w_overlap for metric in best_sources
                    ),
                )
            )
    return records


def write_summary(path: Path, records: list[Record]) -> None:
    fieldnames = [
        "case",
        "current_version",
        "current_timelimit",
        "best_sources",
        "current_objective",
        "best_objective",
        "score_ratio",
        "current_tardiness",
        "best_tardiness",
        "tardiness_improvement",
        "current_n_good",
        "best_n_good",
        "delta_n_good",
        "current_w_large",
        "best_w_large",
        "delta_w_large",
        "current_w_overlap",
        "best_w_overlap",
        "delta_w_overlap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.row
            delta_n_good = record.best_n_good - record.current.n_good
            delta_w_large = record.best_w_large - record.current.w_large
            delta_w_overlap = record.best_w_overlap - record.current.w_overlap
            writer.writerow(
                {
                    **{name: row[name] for name in fieldnames[:10]},
                    "current_n_good": record.current.n_good,
                    "best_n_good": record.best_n_good,
                    "delta_n_good": delta_n_good,
                    "current_w_large": record.current.w_large,
                    "best_w_large": record.best_w_large,
                    "delta_w_large": delta_w_large,
                    "current_w_overlap": record.current.w_overlap,
                    "best_w_overlap": record.best_w_overlap,
                    "delta_w_overlap": delta_w_overlap,
                }
            )


def average_ranks(values: list[float]) -> list[float]:
    indices = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    first = 0
    while first < len(indices):
        end = first + 1
        while end < len(indices) and values[indices[first]] == values[indices[end]]:
            end += 1
        rank = (first + 1 + end) / 2.0
        for index in indices[first:end]:
            ranks[index] = rank
        first = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rank_x = average_ranks(xs)
    rank_y = average_ranks(ys)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    covariance = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(rank_x, rank_y)
    )
    variance_x = sum((x - mean_x) ** 2 for x in rank_x)
    variance_y = sum((y - mean_y) ** 2 for y in rank_y)
    if variance_x == 0.0 or variance_y == 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def format_correlation(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def delta_values(record: Record) -> tuple[float, float, float]:
    return (
        record.best_n_good - record.current.n_good,
        record.best_w_large - record.current.w_large,
        record.best_w_overlap - record.current.w_overlap,
    )


def print_group(label: str, records: list[Record]) -> None:
    deltas = [delta_values(record) for record in records]
    better = [sum(delta[index] > 0.0 for delta in deltas) for index in range(3)]
    print(
        f"{label}: best>current N_good={better[0]}/{len(records)}, "
        f"W_large={better[1]}/{len(records)}, W_overlap={better[2]}/{len(records)}"
    )
    improvements = [
        float(record.row["score_ratio"]) - 1.0 for record in records
    ]
    correlations = [
        format_correlation(
            spearman(improvements, [delta[index] for delta in deltas])
        )
        for index in range(3)
    ]
    print(
        f"{label}: Spearman score_improvement vs "
        f"delta_N_good={correlations[0]}, delta_W_large={correlations[1]}, "
        f"delta_W_overlap={correlations[2]}"
    )


def print_statistics(records: list[Record]) -> None:
    print(f"processed cases: {len(records)}")
    t_zero = [
        record
        for record in records
        if float(record.row["current_tardiness"]) <= 0.0
        and float(record.row["best_tardiness"]) <= 0.0
    ]
    t_positive = [record for record in records if record not in t_zero]
    for label, group in [("all", records), ("T=0", t_zero), ("T>0", t_positive)]:
        print_group(label, group)

    improvements = [
        float(record.row["tardiness_improvement"]) for record in t_positive
    ]
    deltas = [delta_values(record) for record in t_positive]
    correlations = [
        format_correlation(
            spearman(improvements, [delta[index] for delta in deltas])
        )
        for index in range(3)
    ]
    print(
        "T>0: Spearman tardiness_improvement vs "
        f"delta_N_good={correlations[0]}, delta_W_large={correlations[1]}, "
        f"delta_W_overlap={correlations[2]}"
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    details_path = resolve(root, args.details)
    summary_path = resolve(root, args.summary)
    output_path = resolve(root, args.output)

    metrics = load_metrics(root, details_path, args.good_ratio)
    records = load_records(summary_path, metrics)
    write_summary(output_path, records)
    print_statistics(records)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
