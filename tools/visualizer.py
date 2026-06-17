#!/usr/bin/env python3
"""Visualize a runner output directory as a bay-wise Gantt chart."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create gantt.svg from log/{version}/{timelimit}/{testcase}.",
    )
    parser.add_argument("run_dir", help="Run result directory, e.g. log/v1/60/prob_1")
    parser.add_argument("--out", help="Output SVG path. default: {run_dir}/gantt.svg")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_case_path(root: Path, case: str) -> Path:
    path = Path(case)
    if not path.is_absolute():
        path = root / path
    return path


def build_assignments(solution: dict[str, Any]) -> list[dict[str, Any]]:
    assignments: dict[int, dict[str, Any]] = {}
    operations = solution.get("operations", {})

    for time_key, ops in operations.items():
        t = int(time_key)
        for op in ops:
            block_id = int(op["block_id"])
            if op["type"] == "ENTRY":
                assignments.setdefault(block_id, {"block_id": block_id})
                assignments[block_id].update(
                    {
                        "entry": t,
                        "bay_id": int(op["bay_id"]),
                        "x": op.get("x"),
                        "y": op.get("y"),
                        "orient_idx": op.get("orient_idx"),
                    }
                )
            elif op["type"] == "EXIT":
                assignments.setdefault(block_id, {"block_id": block_id})
                assignments[block_id].update(
                    {
                        "exit": t,
                        "exit_bay_id": int(op["bay_id"]),
                    }
                )

    complete = [
        assignment
        for assignment in assignments.values()
        if "entry" in assignment and "exit" in assignment and "bay_id" in assignment
    ]
    complete.sort(key=lambda a: (a["bay_id"], a["entry"], a["exit"], a["block_id"]))
    return complete


def block_metrics(
    prob_info: dict[str, Any], assignment: dict[str, Any]
) -> dict[str, Any]:
    block = prob_info["blocks"][assignment["block_id"]]
    due = int(block["due_date"])
    release = int(block["release_time"])
    processing = int(block["processing_time"])
    entry = int(assignment["entry"])
    exit_time = int(assignment["exit"])
    tardiness = max(0, exit_time - due)
    waiting = max(0, entry - release)
    stay = exit_time - entry
    extra_stay = max(0, stay - processing)
    return {
        "due": due,
        "release": release,
        "processing": processing,
        "tardiness": tardiness,
        "waiting": waiting,
        "stay": stay,
        "extra_stay": extra_stay,
    }


def tardiness_color(tardiness: int) -> str:
    if tardiness == 0:
        return "#86efac"
    if tardiness <= 10:
        return "#fde68a"
    if tardiness <= 50:
        return "#fb923c"
    return "#f87171"


def svg_text(
    x: float, y: float, text: str, size: int = 12, anchor: str = "start"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Menlo, Consolas, monospace" text-anchor="{anchor}" '
        f'fill="#111827">{html.escape(text)}</text>'
    )


def render_svg(
    prob_info: dict[str, Any],
    assignments: list[dict[str, Any]],
    meta: dict[str, Any],
    result: dict[str, Any] | None,
) -> str:
    n_bays = len(prob_info.get("bays", []))
    max_exit = max((int(a["exit"]) for a in assignments), default=1)
    max_time = max(1, max_exit)

    left = 90
    right = 40
    top = 74
    row_h = 34
    chart_w = 1200
    footer_top = top + n_bays * row_h + 58
    footer_h = 260
    width = left + chart_w + right
    height = footer_top + footer_h
    bar_h = 22

    def x_of(t: float) -> float:
        return left + t / max_time * chart_w

    elements: list[str] = []
    elements.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    elements.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    testcase = str(meta.get("testcase", ""))
    version = str(meta.get("version", ""))
    timelimit = meta.get("timelimit", "")
    title = f"version={version} testcase={testcase} timelimit={timelimit}"
    elements.append(svg_text(20, 26, title, size=16))

    if result:
        objective = result.get("objective")
        obj1 = result.get("obj1")
        obj2 = result.get("obj2")
        obj3 = result.get("obj3")
        feasible = result.get("feasible")
        summary = f"feasible={feasible} objective={objective} obj1={obj1} obj2={obj2} obj3={obj3}"
        elements.append(svg_text(20, 50, summary, size=13))

    tick_count = 10
    for i in range(tick_count + 1):
        t = max_time * i / tick_count
        x = x_of(t)
        elements.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{top + n_bays * row_h}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            svg_text(x, top + n_bays * row_h + 20, f"{t:.0f}", size=11, anchor="middle")
        )

    by_bay: dict[int, list[dict[str, Any]]] = {bay_id: [] for bay_id in range(n_bays)}
    for assignment in assignments:
        by_bay.setdefault(int(assignment["bay_id"]), []).append(assignment)

    for bay_id in range(n_bays):
        y = top + bay_id * row_h
        elements.append(svg_text(18, y + 18, f"Bay {bay_id}", size=13))
        elements.append(
            f'<line x1="{left}" y1="{y + row_h - 5}" x2="{left + chart_w}" y2="{y + row_h - 5}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        for assignment in by_bay.get(bay_id, []):
            metrics = block_metrics(prob_info, assignment)
            entry = int(assignment["entry"])
            exit_time = int(assignment["exit"])
            x = x_of(entry)
            w = max(1.0, x_of(exit_time) - x)
            fill = tardiness_color(metrics["tardiness"])
            label = f"B{assignment['block_id']}"
            if metrics["tardiness"] > 0:
                label += f" +{metrics['tardiness']}"
            tooltip = (
                f"block={assignment['block_id']} bay={bay_id} "
                f"entry={entry} exit={exit_time} due={metrics['due']} "
                f"tardiness={metrics['tardiness']}"
            )
            elements.append(
                f'<rect x="{x:.1f}" y="{y + 5:.1f}" width="{w:.1f}" height="{bar_h}" '
                f'rx="4" fill="{fill}" stroke="#374151" stroke-width="0.8">'
                f"<title>{html.escape(tooltip)}</title></rect>"
            )
            if w >= 22:
                elements.append(svg_text(x + 4, y + 20, label, size=11))

    metrics_rows = []
    for assignment in assignments:
        metrics = block_metrics(prob_info, assignment)
        if metrics["tardiness"] > 0:
            metrics_rows.append((metrics["tardiness"], assignment, metrics))
    metrics_rows.sort(key=lambda item: (-item[0], item[1]["exit"], item[1]["block_id"]))

    elements.append(svg_text(20, footer_top, "Tardiness top 10", size=15))
    if not metrics_rows:
        elements.append(svg_text(20, footer_top + 24, "none", size=12))
    else:
        for rank, (tardiness, assignment, metrics) in enumerate(
            metrics_rows[:10], start=1
        ):
            text = (
                f"{rank:2}. B{assignment['block_id']} bay={assignment['bay_id']} "
                f"release={metrics['release']} due={metrics['due']} "
                f"entry={assignment['entry']} exit={assignment['exit']} tardiness={tardiness}"
            )
            elements.append(svg_text(20, footer_top + 24 + rank * 20, text, size=12))

    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def main() -> int:
    args = parse_args()
    root = repo_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    meta_path = run_dir / "meta.json"
    solution_path = run_dir / "solution.json"
    result_path = run_dir / "result.json"
    if not meta_path.is_file():
        print(f"error: {meta_path} not found", file=sys.stderr)
        return 1
    if not solution_path.is_file():
        print(f"error: {solution_path} not found", file=sys.stderr)
        return 1

    meta = load_json(meta_path)
    solution = load_json(solution_path)
    result = load_json(result_path) if result_path.is_file() else None

    case = meta.get("case")
    if not isinstance(case, str) or not case:
        print("error: meta.json does not contain a valid case", file=sys.stderr)
        return 1
    case_path = resolve_case_path(root, case)
    if not case_path.is_file():
        print(f"error: case file not found: {case_path}", file=sys.stderr)
        return 1
    prob_info = load_json(case_path)

    assignments = build_assignments(solution)
    svg = render_svg(prob_info, assignments, meta, result)

    out_path = Path(args.out) if args.out else run_dir / "gantt.svg"
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(f"created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
