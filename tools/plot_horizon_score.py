# ruff: noqa
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.lines import Line2D  # type: ignore


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
HORIZON_RE = re.compile(
    r"^\[(?P<time>[0-9.]+)\] \[optimize\] horizon=(?P<horizon>\d+)/(?P<total>\d+), "
    r"blocks=(?P<blocks>\d+), initial=(?P<initial>[-+0-9.eE]+), w2=(?P<w2>[-+0-9.eE]+)"
)
WORKER_RE = re.compile(
    r"^\[(?P<time>[0-9.]+)\] \[optimize worker=(?P<worker>\d+)\] "
    r"iter=\s*(?P<iter>\d+), current=(?P<current>[-+0-9.eE]+), "
    r"temperature=(?P<temperature>[-+0-9.eE]+)"
)
SHARED_BEST_RE = re.compile(
    r"^\[(?P<time>[0-9.]+)\] \[optimize\] shared best: "
    r"worker=(?P<worker>\d+), iter=\s*(?P<iter>\d+), score=(?P<score>[-+0-9.eE]+)"
)
END_RE = re.compile(r"^\[(?P<time>[0-9.]+)\] \[main\] end\.")


@dataclass
class WorkerPoint:
    time: float
    current: float
    temperature: float


@dataclass
class Horizon:
    index: int
    total: int
    start: float
    blocks: int
    initial: float
    w2: float
    workers: dict[int, list[WorkerPoint]] = field(default_factory=dict)
    shared_best: list[tuple[float, float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot rolling-horizon worker scores and temperatures."
    )
    parser.add_argument(
        "input_path", help="Path to stderr.log or a directory containing */stderr.log"
    )
    parser.add_argument("--out", help="Output image path")
    parser.add_argument(
        "--cols", type=int, default=4, help="Columns in directory mode. default: 4"
    )
    return parser.parse_args()


def parse_log(path: Path) -> tuple[list[Horizon], float | None]:
    horizons: list[Horizon] = []
    current_horizon: Horizon | None = None
    end_time = None

    with path.open(encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = ANSI_RE.sub("", raw_line.strip())
            if match := HORIZON_RE.match(line):
                current_horizon = Horizon(
                    index=int(match.group("horizon")),
                    total=int(match.group("total")),
                    start=float(match.group("time")),
                    blocks=int(match.group("blocks")),
                    initial=float(match.group("initial")),
                    w2=float(match.group("w2")),
                )
                horizons.append(current_horizon)
                continue

            if current_horizon is not None and (match := WORKER_RE.match(line)):
                worker = int(match.group("worker"))
                current_horizon.workers.setdefault(worker, []).append(
                    WorkerPoint(
                        time=float(match.group("time")),
                        current=float(match.group("current")),
                        temperature=float(match.group("temperature")),
                    )
                )
                continue

            if current_horizon is not None and (match := SHARED_BEST_RE.match(line)):
                current_horizon.shared_best.append(
                    (float(match.group("time")), float(match.group("score")))
                )
                continue

            if match := END_RE.match(line):
                end_time = float(match.group("time"))

    return horizons, end_time


def plot_single(
    horizons: list[Horizon], end_time: float | None, out_path: Path
) -> None:
    worker_ids = sorted(
        {worker for horizon in horizons for worker in horizon.workers}
    )
    colors = {
        worker: plt.get_cmap("tab10")(index % 10)
        for index, worker in enumerate(worker_ids)
    }

    figure, (score_ax, temperature_ax) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, constrained_layout=True
    )

    for position, horizon in enumerate(horizons):
        next_start = (
            horizons[position + 1].start
            if position + 1 < len(horizons)
            else end_time
        )
        if position % 2 == 0 and next_start is not None:
            score_ax.axvspan(horizon.start, next_start, color="black", alpha=0.035)
            temperature_ax.axvspan(
                horizon.start, next_start, color="black", alpha=0.035
            )

        for axis in (score_ax, temperature_ax):
            axis.axvline(horizon.start, color="gray", linewidth=0.8, alpha=0.7)

        score_ax.scatter(
            [horizon.start],
            [horizon.initial],
            color="black",
            marker="x",
            s=35,
            zorder=4,
        )
        score_ax.annotate(
            f"H{horizon.index} blocks={horizon.blocks} w2={horizon.w2:g}",
            (horizon.start, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -4),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="dimgray",
        )

        for worker, points in sorted(horizon.workers.items()):
            times = [point.time for point in points]
            score_ax.plot(
                times,
                [point.current for point in points],
                color=colors[worker],
                linewidth=1.2,
                marker=".",
                markersize=3,
                label=f"worker {worker}" if position == 0 else None,
            )
            temperature_ax.plot(
                times,
                [point.temperature for point in points],
                color=colors[worker],
                linewidth=1.2,
                marker=".",
                markersize=3,
                label=f"worker {worker}" if position == 0 else None,
            )

        if horizon.shared_best:
            shared_times = [horizon.start]
            shared_scores = [horizon.initial]
            shared_times.extend(time for time, _ in horizon.shared_best)
            shared_scores.extend(score for _, score in horizon.shared_best)
            if next_start is not None:
                shared_times.append(next_start)
                shared_scores.append(shared_scores[-1])
            score_ax.step(
                shared_times,
                shared_scores,
                where="post",
                color="black",
                linewidth=1.0,
                alpha=0.8,
                label="shared best" if position == 0 else None,
            )

    score_ax.set_title("Rolling-horizon optimize score by worker")
    score_ax.set_ylabel("current score")
    score_ax.grid(True, alpha=0.25)
    temperature_ax.set_ylabel("temperature")
    temperature_ax.set_xlabel("elapsed time [s]")
    temperature_ax.grid(True, alpha=0.25)

    if worker_ids:
        score_ax.legend(loc="best", ncol=min(4, len(worker_ids) + 1))
        temperature_ax.legend(loc="best", ncol=min(4, len(worker_ids)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def natural_key(path: Path) -> list[str | int]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def plot_multiple(
    cases: list[tuple[str, list[Horizon], float | None]],
    out_path: Path,
    cols: int,
) -> None:
    rows = math.ceil(len(cases) / cols)
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 3.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    worker_ids = sorted(
        {
            worker
            for _, horizons, _ in cases
            for horizon in horizons
            for worker in horizon.workers
        }
    )
    colors = {
        worker: plt.get_cmap("tab10")(index % 10)
        for index, worker in enumerate(worker_ids)
    }

    for case_index, (case_name, horizons, end_time) in enumerate(cases):
        score_ax = axes[case_index // cols][case_index % cols]
        temperature_ax = score_ax.twinx()
        for position, horizon in enumerate(horizons):
            next_start = (
                horizons[position + 1].start
                if position + 1 < len(horizons)
                else end_time
            )
            if position % 2 == 0 and next_start is not None:
                score_ax.axvspan(
                    horizon.start, next_start, color="black", alpha=0.035
                )
            score_ax.axvline(
                horizon.start, color="gray", linewidth=0.6, alpha=0.6
            )
            score_ax.scatter(
                [horizon.start],
                [horizon.initial],
                color="black",
                marker="x",
                s=12,
                zorder=4,
            )
            score_ax.annotate(
                f"H{horizon.index}",
                (horizon.start, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="left",
                fontsize=5,
                color="dimgray",
            )

            for worker, points in sorted(horizon.workers.items()):
                times = [point.time for point in points]
                score_ax.plot(
                    times,
                    [point.current for point in points],
                    color=colors[worker],
                    linewidth=0.8,
                )
                temperature_ax.plot(
                    times,
                    [point.temperature for point in points],
                    color=colors[worker],
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.7,
                )

            if horizon.shared_best:
                shared_times = [horizon.start]
                shared_scores = [horizon.initial]
                shared_times.extend(time for time, _ in horizon.shared_best)
                shared_scores.extend(score for _, score in horizon.shared_best)
                if next_start is not None:
                    shared_times.append(next_start)
                    shared_scores.append(shared_scores[-1])
                score_ax.step(
                    shared_times,
                    shared_scores,
                    where="post",
                    color="black",
                    linewidth=0.8,
                    alpha=0.8,
                )

        score_ax.set_title(case_name, fontsize=10)
        score_ax.set_xlabel("elapsed [s]", fontsize=8)
        score_ax.set_ylabel("score", fontsize=8)
        temperature_ax.set_ylabel("temperature", fontsize=8)
        score_ax.tick_params(labelsize=7)
        temperature_ax.tick_params(labelsize=7)
        score_ax.grid(True, alpha=0.2)

    for case_index in range(len(cases), rows * cols):
        axes[case_index // cols][case_index % cols].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=colors[worker], label=f"worker {worker}")
        for worker in worker_ids
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color="black", label="shared best"),
            Line2D([0], [0], color="gray", linestyle="--", label="temperature"),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=6,
    )
    figure.suptitle("Rolling-horizon scores and temperatures", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    if input_path.is_file():
        horizons, end_time = parse_log(input_path)
        if not horizons:
            print(
                f"error: no optimize horizon logs found in {input_path}",
                file=sys.stderr,
            )
            return 1
        out_path = (
            Path(args.out).resolve()
            if args.out
            else input_path.with_name("horizon-score.png")
        )
        plot_single(horizons, end_time, out_path)
    elif input_path.is_dir():
        cases = []
        for log_path in sorted(input_path.glob("*/stderr.log"), key=natural_key):
            horizons, end_time = parse_log(log_path)
            if not horizons:
                print(
                    f"warning: no optimize horizon logs found in {log_path}",
                    file=sys.stderr,
                )
                continue
            cases.append((log_path.parent.name, horizons, end_time))
        if not cases:
            print(f"error: no usable stderr.log under {input_path}", file=sys.stderr)
            return 1
        out_path = (
            Path(args.out).resolve()
            if args.out
            else input_path / "horizon-scores.png"
        )
        plot_multiple(cases, out_path, args.cols)
    else:
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    print(f"created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
