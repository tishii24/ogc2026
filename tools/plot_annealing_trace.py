#!/usr/bin/env python3
"""Render trace-annealing CSV files as a standalone HTML report."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter
from pathlib import Path
from typing import Callable

COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
]
SIGNIFICANT_EVENTS = {"exchange", "local_best", "shared_best"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=Path("annealing-trace.html"))
    return parser.parse_args()


def parse_float(value: str) -> float | None:
    if not value:
        return None
    return float(value)


def load_traces(trace_dir: Path) -> dict[str, list[dict[str, str]]]:
    traces: dict[str, list[dict[str, str]]] = {}
    for path in sorted(trace_dir.glob("global*-worker-*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            traces[path.stem] = rows
    return traces


def downsample(rows: list[dict[str, str]], limit: int = 4000) -> list[dict[str, str]]:
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    selected = [rows[min(int(index * step), len(rows) - 1)] for index in range(limit)]
    if selected[-1] is not rows[-1]:
        selected[-1] = rows[-1]
    return selected


def make_plot(
    traces: dict[str, list[dict[str, str]]],
    title: str,
    value: Callable[[dict[str, str]], float | None],
    value_label: str,
    events: set[str] | None = None,
) -> str:
    width = 1200
    height = 430
    left = 72
    right = 24
    top = 34
    bottom = 52
    plot_width = width - left - right
    plot_height = height - top - bottom

    points_by_trace: dict[str, list[tuple[float, float, dict[str, str]]]] = {}
    for name, rows in traces.items():
        points = []
        filtered = [row for row in rows if events is None or row["event"] in events]
        for row in downsample(filtered):
            y = value(row)
            x = parse_float(row["elapsed"])
            if x is not None and y is not None and math.isfinite(y):
                points.append((x, y, row))
        if points:
            points_by_trace[name] = points

    all_points = [point for points in points_by_trace.values() for point in points]
    if not all_points:
        return f"<section><h2>{html.escape(title)}</h2><p>no data</p></section>"
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0
    y_margin = (max_y - min_y) * 0.04
    min_y -= y_margin
    max_y += y_margin

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_width

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_height

    svg = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fff" stroke="#cbd5e1"/>',
    ]
    for tick in range(6):
        ratio = tick / 5
        x = left + ratio * plot_width
        elapsed = min_x + ratio * (max_x - min_x)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle">{elapsed:.1f}</text>')
        y = top + ratio * plot_height
        y_value = max_y - ratio * (max_y - min_y)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{y_value:.3g}</text>')
    svg.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 4}" text-anchor="middle">elapsed seconds</text>')
    svg.append(f'<text transform="translate(16 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(value_label)}</text>')

    legend = []
    for index, (name, points) in enumerate(points_by_trace.items()):
        color = COLORS[index % len(COLORS)]
        polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in points)
        svg.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.4" opacity="0.85"/>')
        for x, y, row in points:
            if row["event"] not in SIGNIFICANT_EVENTS:
                continue
            tooltip = (
                f'{name} event={row["event"]} t={x:.3f} iter={row["iteration"]} '
                f'before={row["before_score"]} after={row["after_score"]} '
                f'z1={row["after_z1"]} changed={row["changed_blocks"]} bay={row["bay_changes"]}'
            )
            svg.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}" stroke="#fff"><title>{html.escape(tooltip)}</title></circle>'
            )
        legend.append(f'<span><i style="background:{color}"></i>{html.escape(name)}</span>')
    svg.append("</svg>")
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f'<div class="legend">{"".join(legend)}</div>'
        f'<div class="chart">{"".join(svg)}</div></section>'
    )


def render_summary(traces: dict[str, list[dict[str, str]]]) -> str:
    headers = ["trace", "events", "accepted", "exchange", "local_best", "shared_best"]
    rows = []
    for name, events in traces.items():
        counts = Counter(row["event"] for row in events)
        rows.append(
            [
                name,
                str(len(events)),
                str(counts["accepted"]),
                str(counts["exchange"]),
                str(counts["local_best"]),
                str(counts["shared_best"]),
            ]
        )
    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    row_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<section><h2>Summary</h2><table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table></section>"


def render_html(traces: dict[str, list[dict[str, str]]]) -> str:
    score_plot = make_plot(
        traces,
        "Current score",
        lambda row: parse_float(row["after_score"]),
        "score",
        {"start", "accepted", "exchange", "finish", "local_best", "shared_best"},
    )
    z1_plot = make_plot(
        traces,
        "Tardiness (log10(z1 + 1))",
        lambda row: math.log10(float(row["after_z1"]) + 1.0),
        "log10(z1 + 1)",
        {"start", "accepted", "exchange", "finish"},
    )
    changed_plot = make_plot(
        traces,
        "Changed blocks per accepted transition",
        lambda row: parse_float(row["changed_blocks"]),
        "changed blocks",
        {"accepted"},
    )
    bay_plot = make_plot(
        traces,
        "Bay changes per accepted transition",
        lambda row: parse_float(row["bay_changes"]),
        "bay changes",
        {"accepted"},
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annealing trace</title>
<style>
body {{ margin: 24px; color: #1f2937; background: #f8fafc; font-family: system-ui, sans-serif; }}
h1, h2 {{ margin: 0 0 12px; }}
section {{ margin: 0 0 24px; padding: 16px; border: 1px solid #dbe3ec; border-radius: 8px; background: #fff; }}
.chart {{ overflow-x: auto; }}
svg {{ display: block; min-width: 900px; width: 100%; font-size: 11px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; font-size: 12px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
.legend i {{ width: 14px; height: 3px; display: inline-block; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 6px 10px; border: 1px solid #dbe3ec; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>Annealing trace</h1>
{render_summary(traces)}
{score_plot}
{z1_plot}
{changed_plot}
{bay_plot}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    traces = load_traces(args.trace_dir)
    if not traces:
        raise SystemExit(f"no trace CSV files found in {args.trace_dir}")
    args.output.write_text(render_html(traces), encoding="utf-8")
    print(f"created: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
