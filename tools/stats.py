#!/usr/bin/env python3
"""Summarize log/score.csv by version."""

from __future__ import annotations

import argparse
import csv
import glob
import json
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
    parser.add_argument(
        "--suite",
        help='Only summarize cases in suite JSON. Format: {"cases": ["train/prob_1.json", ...]}',
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def normalize_case_path(root: Path, case: str) -> str:
    path = Path(case)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.is_relative_to(root):
        return str(path.relative_to(root))
    return str(path)


def load_suite_cases(root: Path, suite: str) -> list[str]:
    path = Path(suite)
    if not path.is_absolute():
        path = root / path
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("suite must be a JSON object")
    patterns = data.get("cases")
    if not isinstance(patterns, list) or not all(
        isinstance(pattern, str) for pattern in patterns
    ):
        raise ValueError("suite.cases must be a list of strings")

    cases: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = (
            glob.glob(str(root / pattern))
            if not Path(pattern).is_absolute()
            else glob.glob(pattern)
        )
        matches.sort(key=lambda path: natural_key(str(Path(path))))
        if matches:
            normalized = [normalize_case_path(root, match) for match in matches]
        else:
            normalized = [normalize_case_path(root, pattern)]
        for case in normalized:
            if case not in seen:
                cases.append(case)
                seen.add(case)
    return cases


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
    return str(round(value))


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

        objectives = [
            objective
            for row in feasible_rows
            if (objective := parse_float(row.get("objective", ""))) is not None
        ]
        best_objective = min(objectives)
        for row in feasible_rows:
            objective = parse_float(row.get("objective", ""))
            if objective is not None and objective == best_objective:
                version = row.get("version", "")
                best_counts[(version, timelimit)] += 1

    return best_counts


def row_key(row: dict[str, str]) -> tuple[str, float] | None:
    version = row.get("version", "")
    timelimit = parse_float(row.get("timelimit", ""))
    if not version or timelimit is None:
        return None
    return (version, timelimit)


def load_case_weights(root: Path, case: str) -> tuple[float, float, float]:
    path = Path(case)
    if not path.is_absolute():
        path = root / path
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    weights = data.get("weights", {})
    return (
        float(weights.get("w1", 1.0)),
        float(weights.get("w2", 1.0)),
        float(weights.get("w3", 1.0)),
    )


def compute_rank_scores(
    rows: list[dict[str, str]], cases: list[str] | None = None
) -> dict[tuple[str, float], int]:
    row_keys = sorted(
        {key for row in rows if (key := row_key(row)) is not None},
        key=lambda key: (natural_key(key[0]), key[1]),
    )
    if cases is None:
        cases = sorted(
            {row.get("case", "") for row in rows if row.get("case", "")},
            key=lambda case: natural_key(case_label(case)),
        )
    if not row_keys or not cases:
        return {}

    by_key = {
        (key[0], key[1], row.get("case", "")): row
        for row in rows
        if (key := row_key(row)) is not None and row.get("case", "")
    }
    rank_scores = {key: 0 for key in row_keys}
    failed_rank = len(row_keys) + 1

    for case in cases:
        feasible: list[tuple[float, tuple[str, float]]] = []
        failed_keys: list[tuple[str, float]] = []
        for key in row_keys:
            row = by_key.get((key[0], key[1], case))
            objective = parse_float(row.get("objective", "")) if row else None
            if row and parse_bool(row.get("feasible", "")) and objective is not None:
                feasible.append((objective, key))
            else:
                failed_keys.append(key)

        feasible.sort(key=lambda item: (item[0], natural_key(item[1][0]), item[1][1]))
        index = 0
        while index < len(feasible):
            objective = feasible[index][0]
            rank = index + 1
            while index < len(feasible) and feasible[index][0] == objective:
                rank_scores[feasible[index][1]] += rank
                index += 1

        for key in failed_keys:
            rank_scores[key] += failed_rank

    return rank_scores


def summarize(
    root: Path, rows: list[dict[str, str]], cases: list[str] | None = None
) -> list[dict[str, Any]]:
    best_counts = compute_best_counts(rows)
    rank_scores = compute_rank_scores(rows, cases)
    groups: dict[tuple[str, float], dict[str, Any]] = {}
    weights_by_case: dict[str, tuple[float, float, float]] = {}

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
                "rank_score": 0,
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
            objective = parse_float(row.get("objective", ""))
            if objective is not None:
                group["total_objective"] += objective

            case = row.get("case", "")
            if case not in weights_by_case:
                weights_by_case[case] = load_case_weights(root, case)
            w1, w2, w3 = weights_by_case[case]
            for src, dst, weight in [
                ("obj1", "total_obj1", w1),
                ("obj2", "total_obj2", w2),
                ("obj3", "total_obj3", w3),
            ]:
                value = parse_float(row.get(src, ""))
                if value is not None:
                    group[dst] += weight * value
        else:
            group["failed"] += 1

    for key, group in groups.items():
        group["best"] = best_counts.get(key, 0)
        group["rank_score"] = rank_scores.get(key, 0)

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
        "rank_score",
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
                str(item["rank_score"]),
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


def print_score_matrix(
    rows: list[dict[str, str]], cases: list[str] | None = None
) -> None:
    if cases is None:
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
    root = repo_root()
    suite_cases: list[str] | None = None
    if args.suite:
        try:
            suite_cases = load_suite_cases(root, args.suite)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    score_path = root / "log" / "score.csv"
    if not score_path.is_file():
        print(f"error: {score_path} not found", file=sys.stderr)
        return 1

    with score_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = latest_rows(rows)
    if suite_cases is not None:
        suite_case_set = set(suite_cases)
        rows = [row for row in rows if row.get("case", "") in suite_case_set]
    if not rows:
        print("error: no valid rows found", file=sys.stderr)
        return 1

    if args.matrix:
        print_score_matrix(rows, suite_cases)
    else:
        print_table(summarize(root, rows, suite_cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
