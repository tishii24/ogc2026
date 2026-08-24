#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore

PALETTE = ["#4389DC", "#D78329", "#A15D99", "#789180", "#716F9C"]
INK = "#3F3F3F"
GRID = "#AEB4BB"
CAPACITY_COLOR = "#D65A35"
CAPACITY = 100
BLOCK_COUNT = 20
HORIZON_COUNT = 4


@dataclass
class Block:
    block_id: int
    release: int
    duration: int
    occupancy: int
    start: int = 0

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass(frozen=True)
class ScheduleData:
    blocks: list[Block]
    admission_order: list[int]
    utilization: np.ndarray


def generate_schedule() -> ScheduleData:
    rng = np.random.default_rng(2026)
    blocks = []
    for block_id in range(BLOCK_COUNT):
        release = max(0, block_id + int(rng.integers(-3, 6)))
        duration = int(rng.integers(6, 14))
        occupancy = int(rng.integers(14, 31))
        blocks.append(Block(block_id, release, duration, occupancy))

    utilization = np.zeros(256, dtype=int)
    scheduling_order = sorted(
        range(BLOCK_COUNT),
        key=lambda block_id: (
            blocks[block_id].release + blocks[block_id].duration,
            -(blocks[block_id].occupancy * blocks[block_id].duration),
            block_id,
        ),
    )
    for block_id in scheduling_order:
        block = blocks[block_id]
        start = block.release
        while np.any(utilization[start : start + block.duration] + block.occupancy > CAPACITY):
            start += 1
        block.start = start
        utilization[start : start + block.duration] += block.occupancy

    admission_order = sorted(
        range(BLOCK_COUNT),
        key=lambda block_id: (
            blocks[block_id].start,
            -(blocks[block_id].occupancy * blocks[block_id].duration),
            block_id,
        ),
    )
    time_max = max(block.end for block in blocks) + 2
    utilization = utilization[:time_max]
    assert utilization.max() <= CAPACITY
    return ScheduleData(blocks, admission_order, utilization)


def draw_gantt(axis: plt.Axes, data: ScheduleData, time_max: int) -> None:
    for row, block_id in enumerate(data.admission_order):
        block = data.blocks[block_id]
        axis.barh(
            row,
            block.duration,
            left=block.start,
            height=0.72,
            color=PALETTE[block_id % len(PALETTE)],
            edgecolor=INK,
            linewidth=0.9,
        )

    for boundary in range(5, BLOCK_COUNT, 5):
        axis.axhline(boundary - 0.5, color=INK, linewidth=1.0, linestyle=(0, (3, 4)), alpha=0.55)

    axis.set_xlim(-1, time_max)
    axis.set_ylim(-0.8, BLOCK_COUNT - 0.2)
    axis.invert_yaxis()
    axis.set_yticks(range(BLOCK_COUNT))
    axis.set_yticklabels([f"B{block_id + 1}" for block_id in data.admission_order], fontsize=7.5)
    axis.tick_params(axis="x", labelbottom=False, length=0)
    axis.tick_params(axis="y", length=0, pad=4)
    axis.grid(axis="x", color=GRID, linewidth=0.8, linestyle=(0, (4, 5)))
    axis.set_axisbelow(True)

    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(False)


def draw_utilization(axis: plt.Axes, data: ScheduleData, time_max: int) -> None:
    times = np.arange(time_max + 1)
    contributions = np.zeros((BLOCK_COUNT, time_max + 1), dtype=float)
    for block_id in data.admission_order:
        block = data.blocks[block_id]
        contributions[block_id, block.start : block.end] = block.occupancy

    axis.stackplot(
        times,
        contributions,
        step="post",
        colors=[PALETTE[index % len(PALETTE)] for index in range(BLOCK_COUNT)],
        edgecolor=INK,
        linewidth=0.35,
    )
    axis.axhline(
        CAPACITY,
        color=CAPACITY_COLOR,
        linewidth=2.0,
        linestyle=(0, (3, 3)),
        label="capacity",
    )
    axis.text(
        time_max + 0.3,
        CAPACITY,
        "capacity",
        color=CAPACITY_COLOR,
        fontsize=10,
        fontweight="bold",
        va="center",
    )
    axis.set_xlim(-1, time_max)
    axis.set_ylim(0, CAPACITY * 1.12)
    axis.set_xlabel("t", fontsize=12, fontweight="bold", labelpad=2)
    axis.set_ylabel("occupancy", fontsize=10, labelpad=5)
    axis.grid(axis="x", color=GRID, linewidth=0.8, linestyle=(0, (4, 5)))
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=8, length=0)
    axis.tick_params(axis="y", labelsize=8, length=0)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)


def draw_horizons(axis: plt.Axes, data: ScheduleData) -> None:
    axis.set_xlim(0, 3)
    axis.set_ylim(0, 4)
    axis.axis("off")

    blocks_per_horizon = BLOCK_COUNT // HORIZON_COUNT
    for horizon in range(HORIZON_COUNT):
        y0 = 3 - horizon + 0.1
        axis.add_patch(
            Rectangle(
                (0.02, y0),
                2.9,
                0.8,
                facecolor="#FFFFFF",
                edgecolor="#C7CBD1",
                linewidth=1.0,
            )
        )
        axis.text(
            0.12,
            y0 + 0.43,
            f"Horizon {horizon + 1}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        block_ids = data.admission_order[
            horizon * blocks_per_horizon : (horizon + 1) * blocks_per_horizon
        ]
        for index, block_id in enumerate(block_ids):
            block = data.blocks[block_id]
            height = 0.3 + 0.24 * block.occupancy / CAPACITY
            x = 1.05 + index * 0.35
            axis.add_patch(
                Rectangle(
                    (x, y0 + 0.26),
                    0.24,
                    height,
                    facecolor=PALETTE[block_id % len(PALETTE)],
                    edgecolor=INK,
                    linewidth=0.8,
                )
            )
            axis.text(x + 0.12, y0 + 0.2, f"B{block_id + 1}", ha="center", va="top", fontsize=6.5)


def main() -> None:
    output = Path(__file__).resolve().parent / "figures/relaxed-schedule.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_schedule()
    time_max = len(data.utilization) - 1

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    figure = plt.figure(figsize=(15.5, 5.4), dpi=240)
    outer = GridSpec(1, 2, figure=figure, width_ratios=[2.15, 1.0], wspace=0.16)
    left = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[1.65, 1.0], hspace=0.22)
    gantt = figure.add_subplot(left[0])
    utilization = figure.add_subplot(left[1])
    horizons = figure.add_subplot(outer[1])

    draw_gantt(gantt, data, time_max)
    draw_utilization(utilization, data, time_max)
    draw_horizons(horizons, data)

    figure.subplots_adjust(left=0.055, right=0.975, bottom=0.12, top=0.94)
    figure.savefig(output, dpi=240, transparent=True)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
