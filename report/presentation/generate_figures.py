#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

CANVAS_WIDTH = 1440
CANVAS_HEIGHT = 800
ORIGIN_X = 110.0
ORIGIN_Y = 720.0
T_SCALE = 17.0
X_SCALE_X = 7.0
X_SCALE_Y = 4.2
Y_SCALE = 18.0

INK = "#536176"
GRID = "#B8C3D1"
BAY_FILL = "#F7F9FC"
PALETTE = ["#AFCBFF", "#C8B6FF", "#B9E8D0", "#FFD0C7", "#FFE69A", "#BDE0FE"]


@dataclass(frozen=True)
class BlockSpec:
    block_id: int
    x: float
    y: float
    entry: float
    exit: float
    color: str


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate presentation figures.")
    parser.add_argument("--input", type=Path, default=root / "in/train/prob_1.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "figures/packing-3d.svg",
    )
    return parser.parse_args()


def project(t: float, x: float, y: float) -> tuple[float, float]:
    return (
        ORIGIN_X + t * T_SCALE + x * X_SCALE_X,
        ORIGIN_Y - y * Y_SCALE - x * X_SCALE_Y,
    )


def points_attr(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def polygons(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    return list(geometry.geoms)


def footprint(block: dict, orientation_index: int = 0) -> Polygon | MultiPolygon:
    layers = []
    for vertices in block["shape"][orientation_index]["layers"]:
        polygon = Polygon(vertices)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            layers.append(polygon)
    geometry = unary_union(layers)
    min_x, min_y, _, _ = geometry.bounds
    return affinity.translate(geometry, xoff=-min_x, yoff=-min_y)


def adjust_color(color: str, factor: float) -> str:
    channels = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    adjusted = [max(0, min(255, round(channel * factor))) for channel in channels]
    return "#" + "".join(f"{channel:02X}" for channel in adjusted)


def geometry_path(geometry: Polygon | MultiPolygon, t: float, x0: float, y0: float) -> str:
    commands = []
    for polygon in polygons(geometry):
        for ring in [polygon.exterior, *polygon.interiors]:
            projected = [project(t, x0 + x, y0 + y) for x, y in ring.coords[:-1]]
            if not projected:
                continue
            commands.append(f"M {projected[0][0]:.2f} {projected[0][1]:.2f}")
            commands.extend(f"L {x:.2f} {y:.2f}" for x, y in projected[1:])
            commands.append("Z")
    return " ".join(commands)


def render_block(spec: BlockSpec, geometry: Polygon | MultiPolygon) -> str:
    side_faces = []
    for polygon in polygons(geometry):
        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = list(ring.coords)
            for start, end in zip(coordinates, coordinates[1:]):
                x1, y1 = start
                x2, y2 = end
                face = [
                    project(spec.entry, spec.x + x1, spec.y + y1),
                    project(spec.exit, spec.x + x1, spec.y + y1),
                    project(spec.exit, spec.x + x2, spec.y + y2),
                    project(spec.entry, spec.x + x2, spec.y + y2),
                ]
                depth = spec.x + (x1 + x2) / 2
                upward = y2 - y1
                factor = 0.91 if upward >= 0 else 0.80
                side_faces.append((depth, face, adjust_color(spec.color, factor)))

    elements = [f'<g id="block-{spec.block_id}" stroke="{INK}" stroke-width="1.8" stroke-linejoin="round">']
    elements.append(f"<title>Block {spec.block_id}</title>")
    for _, face, color in sorted(side_faces, key=lambda item: item[0], reverse=True):
        elements.append(
            f'<polygon points="{points_attr(face)}" fill="{color}" fill-opacity="0.96" stroke="{INK}" stroke-width="1.8"/>'
        )

    entry_path = geometry_path(geometry, spec.entry, spec.x, spec.y)
    exit_path = geometry_path(geometry, spec.exit, spec.x, spec.y)
    elements.append(
        f'<path d="{entry_path}" fill="{adjust_color(spec.color, 0.86)}" fill-rule="evenodd" stroke="{INK}" stroke-width="1.8"/>'
    )
    elements.append(
        f'<path d="{exit_path}" fill="{spec.color}" fill-rule="evenodd" stroke="{INK}" stroke-width="1.8"/>'
    )
    elements.append("</g>")
    return "\n".join(elements)


def line(a: tuple[float, float], b: tuple[float, float], **attrs: str) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" {attributes}/>'


def render_bay(max_t: float, width: float, height: float) -> str:
    corners = {
        (t, x, y): project(t, x, y)
        for t in (0.0, max_t)
        for x in (0.0, width)
        for y in (0.0, height)
    }
    elements = ['<g id="bay" fill="none">']
    floor = [corners[(0.0, 0.0, 0.0)], corners[(max_t, 0.0, 0.0)], corners[(max_t, width, 0.0)], corners[(0.0, width, 0.0)]]
    elements.append(
        f'<polygon points="{points_attr(floor)}" fill="{BAY_FILL}" fill-opacity="0.72" stroke="none"/>'
    )

    for t in range(10, int(max_t), 10):
        elements.append(
            line(project(t, 0, 0), project(t, width, 0), stroke=GRID, stroke_width="1", stroke_opacity="0.45")
        )
    for x in range(10, int(width), 10):
        elements.append(
            line(project(0, x, 0), project(max_t, x, 0), stroke=GRID, stroke_width="1", stroke_opacity="0.35")
        )

    for x in (0.0, width):
        for y in (0.0, height):
            elements.append(
                line(corners[(0.0, x, y)], corners[(max_t, x, y)], stroke=GRID, stroke_width="1.5", stroke_dasharray="6 7")
            )
    for t in (0.0, max_t):
        for y in (0.0, height):
            elements.append(
                line(corners[(t, 0.0, y)], corners[(t, width, y)], stroke=GRID, stroke_width="1.5", stroke_dasharray="6 7")
            )
        for x in (0.0, width):
            elements.append(
                line(corners[(t, x, 0.0)], corners[(t, x, height)], stroke=GRID, stroke_width="1.5", stroke_dasharray="6 7")
            )
    elements.append("</g>")
    return "\n".join(elements)


def render_axes() -> str:
    origin = (92.0, 265.0)
    t_end = (235.0, 265.0)
    x_end = (165.0, 221.0)
    y_end = (92.0, 125.0)
    return "\n".join(
        [
            '<g id="axes" stroke="none">',
            f'<polygon points="92,263.7 224,263.7 224,266.3 92,266.3" fill="{INK}"/>',
            f'<polygon points="90.7,265 90.7,136 93.3,136 93.3,265" fill="{INK}"/>',
            f'<polygon points="91.3,263.9 157.3,224.0 158.7,226.2 92.7,266.1" fill="{INK}"/>',
            f'<polygon points="235,265 222,257 222,273" fill="{INK}"/>',
            f'<polygon points="165,221 151,222 158,234" fill="{INK}"/>',
            f'<polygon points="92,125 84,138 100,138" fill="{INK}"/>',
            "</g>",
            '<g id="axis-labels" fill="{0}" font-family="Inter, Arial, sans-serif" font-size="28" font-style="italic" font-weight="600">'.format(INK),
            f'<text x="{t_end[0] + 12:.2f}" y="{t_end[1] + 8:.2f}">t</text>',
            f'<text x="{x_end[0] + 10:.2f}" y="{x_end[1] - 4:.2f}">x</text>',
            f'<text x="{y_end[0] - 7:.2f}" y="{y_end[1] - 14:.2f}">y</text>',
            "</g>",
        ]
    )


def render(problem: dict) -> str:
    specs = [
        BlockSpec(0, 2, 1, 0, 10, PALETTE[0]),
        BlockSpec(5, 13, 1, 3, 13, PALETTE[1]),
        BlockSpec(9, 31, 1, 6, 14, PALETTE[2]),
        BlockSpec(11, 23, 12, 11, 16, PALETTE[3]),
        BlockSpec(6, 40, 12, 16, 22, PALETTE[4]),
        BlockSpec(1, 4, 1, 25, 35, PALETTE[5]),
    ]
    block_elements = []
    for spec in sorted(specs, key=lambda item: item.x, reverse=True):
        geometry = footprint(problem["blocks"][spec.block_id])
        block_elements.append(render_block(spec, geometry))

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',

            render_bay(55, problem["bays"][0]["width"], problem["bays"][0]["height"]),
            '<g id="blocks">',
            *block_elements,
            "</g>",
            render_axes(),
            "</svg>",
        ]
    )


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as file:
        problem = json.load(file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(problem), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
