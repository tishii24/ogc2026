#!/usr/bin/env python3
"""Summarize log/score.csv by version."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize log/score.csv.")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Show a version/timelimit x testcase score matrix.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_float(value: str) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def format_number(value: float) -> str:
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return f"{value:.3f}"


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def case_label(case: str) -> str:
    stem = Path(case).stem
    match = re.fullmatch(r"prob_(\d+)", stem)
    if match:
        return match.group(1)
    return stem


def latest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str, float], tuple[str, int, dict[str, str]]] = {}
    for index, row in enumerate(rows):
        version = row.get("version", "")
        case = row.get("case", "")
        timelimit = parse_float(row.get("timelimit", ""))
        timestamp = row.get("timestamp", "")
        if not version or not case or timelimit is None:
            continue

        key = (version, case, timelimit)
        current = latest.get(key)
        if current is None or (timestamp, index) > (current[0], current[1]):
            latest[key] = (timestamp, index, row)

    return [item[2] for item in latest.values()]


def compute_best_counts(rows: list[dict[str, str]]) -> dict[tuple[str, float], int]:
    by_case: dict[tuple[float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        timelimit = parse_float(row.get("timelimit", ""))
        case = row.get("case", "")
        if timelimit is None or not case:
            continue
        by_case[(timelimit, case)].append(row)

    best_counts: dict[tuple[str, float], int] = defaultdict(int)
    for (timelimit, _case), case_rows in by_case.items():
        feasible_rows = [
            row
            for row in case_rows
            if parse_bool(row.get("feasible", ""))
            and parse_float(row.get("objective", "")) is not None
        ]
        if not feasible_rows:
            continue

        best_objective = min(
            parse_float(row.get("objective", "")) for row in feasible_rows
        )
        assert best_objective is not None
        for row in feasible_rows:
            objective = parse_float(row.get("objective", ""))
            if objective is not None and objective == best_objective:
                version = row.get("version", "")
                best_counts[(version, timelimit)] += 1

    return best_counts


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    best_counts = compute_best_counts(rows)
    groups: dict[tuple[str, float], dict[str, Any]] = {}

    for row in rows:
        version = row.get("version", "")
        timelimit = parse_float(row.get("timelimit", ""))
        if not version or timelimit is None:
            continue

        key = (version, timelimit)
        group = groups.setdefault(
            key,
            {
                "version": version,
                "timelimit": timelimit,
                "cases": 0,
                "feasible": 0,
                "failed": 0,
                "best": 0,
                "total_objective": 0.0,
                "total_obj1": 0.0,
                "total_obj2": 0.0,
                "total_obj3": 0.0,
                "total_elapsed": 0.0,
            },
        )

        group["cases"] += 1
        elapsed = parse_float(row.get("elapsed", ""))
        if elapsed is not None:
            group["total_elapsed"] += elapsed

        if parse_bool(row.get("feasible", "")):
            group["feasible"] += 1
            for src, dst in [
                ("objective", "total_objective"),
                ("obj1", "total_obj1"),
                ("obj2", "total_obj2"),
                ("obj3", "total_obj3"),
            ]:
                value = parse_float(row.get(src, ""))
                if value is not None:
                    group[dst] += value
        else:
            group["failed"] += 1

    for key, group in groups.items():
        group["best"] = best_counts.get(key, 0)

    return sorted(groups.values(), key=lambda g: g["version"])


def print_rows(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.rjust(widths[index]) for index, cell in enumerate(row)))


def print_table(summaries: list[dict[str, Any]]) -> None:
    headers = [
        "version",
        "tl",
        "cases",
        "feasible",
        "failed",
        "best",
        "total_objective",
        "total_obj1",
        "total_obj2",
        "total_obj3",
        "total_elapsed",
    ]
    rows = []
    for item in summaries:
        rows.append(
            [
                item["version"],
                format_number(item["timelimit"]),
                str(item["cases"]),
                str(item["feasible"]),
                str(item["failed"]),
                str(item["best"]),
                format_number(item["total_objective"]),
                format_number(item["total_obj1"]),
                format_number(item["total_obj2"]),
                format_number(item["total_obj3"]),
                f"{item['total_elapsed']:.3f}",
            ]
        )

    print_rows(headers, rows)


def score_cell(row: dict[str, str] | None) -> str:
    if row is None:
        return "-"
    objective = parse_float(row.get("objective", ""))
    if not parse_bool(row.get("feasible", "")) or objective is None:
        return "NG"
    return format_number(objective)


def print_score_matrix(rows: list[dict[str, str]]) -> None:
    cases = sorted(
        {row.get("case", "") for row in rows if row.get("case", "")},
        key=lambda case: natural_key(case_label(case)),
    )
    row_keys = sorted(
        {
            (row.get("version", ""), parse_float(row.get("timelimit", "")))
            for row in rows
            if row.get("version", "")
            and parse_float(row.get("timelimit", "")) is not None
        },
        key=lambda key: (natural_key(key[0]), key[1]),
    )
    by_key = {
        (
            row.get("version", ""),
            parse_float(row.get("timelimit", "")),
            row.get("case", ""),
        ): row
        for row in rows
    }

    headers = ["version", "tl"] + [case_label(case) for case in cases]
    table_rows = []
    for version, timelimit in row_keys:
        assert timelimit is not None
        table_rows.append(
            [version, format_number(timelimit)]
            + [score_cell(by_key.get((version, timelimit, case))) for case in cases]
        )

    print_rows(headers, table_rows)


def main() -> int:
    args = parse_args()
    score_path = repo_root() / "log" / "score.csv"
    if not score_path.is_file():
        print(f"error: {score_path} not found", file=sys.stderr)
        return 1

    with score_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = latest_rows(rows)
    if not rows:
        print("error: no valid rows found", file=sys.stderr)
        return 1

    if args.matrix:
        print_score_matrix(rows)
    else:
        print_table(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
