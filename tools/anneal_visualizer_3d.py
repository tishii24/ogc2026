# ruff: noqa
# type: ignore
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate, unary_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3D annealing viewer from worker JSONL snapshots."
    )
    parser.add_argument("problem", help="Problem JSON path")
    parser.add_argument("visualize_dir", help="Directory containing worker_{id}.jsonl")
    parser.add_argument("--worker-id", type=int, required=True, help="Worker id to visualize")
    parser.add_argument(
        "--out", help="Output HTML path. default: {visualize_dir}/worker_{id}_3d.html"
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved.resolve()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    snapshots = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if item.get("type") == "snapshot":
                snapshots.append(item)
    snapshots.sort(key=lambda item: (float(item.get("elapsed", 0)), int(item.get("iter", 0))))
    return snapshots


def block_color(block_id: int) -> str:
    hue = (block_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.48, 0.88)
    return f"rgb({round(red * 255)},{round(green * 255)},{round(blue * 255)})"


def orientation_union(orientation: dict[str, Any]) -> Polygon | MultiPolygon:
    polygons = []
    for layer in orientation.get("layers", []):
        if len(layer) >= 3:
            polygon = Polygon([(float(point[0]), float(point[1])) for point in layer])
            if not polygon.is_empty:
                polygons.append(polygon)
    if not polygons:
        return Polygon()
    geometry = unary_union(polygons)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    polygon_parts = [part for part in geometry.geoms if isinstance(part, Polygon)]
    return unary_union(polygon_parts) if polygon_parts else Polygon()


def polygon_parts(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    return list(geometry.geoms)


@dataclass
class Mesh:
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    i: list[int] = field(default_factory=list)
    j: list[int] = field(default_factory=list)
    k: list[int] = field(default_factory=list)
    facecolor: list[str] = field(default_factory=list)
    hovertext: list[str] = field(default_factory=list)

    def add_triangle(
        self,
        points: list[tuple[float, float, float]],
        color: str,
        hover: str,
    ) -> None:
        start = len(self.x)
        for x, y, z in points:
            self.x.append(x)
            self.y.append(y)
            self.z.append(z)
            self.hovertext.append(hover)
        self.i.append(start)
        self.j.append(start + 1)
        self.k.append(start + 2)
        self.facecolor.append(color)


def add_extruded_polygon(
    mesh: Mesh,
    polygon: Polygon,
    entry_time: float,
    exit_time: float,
    color: str,
    hover: str,
) -> None:
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle):
            continue
        coordinates = list(triangle.exterior.coords)[:3]
        bottom = [(x, y, entry_time) for x, y in reversed(coordinates)]
        top = [(x, y, exit_time) for x, y in coordinates]
        mesh.add_triangle(bottom, color, hover)
        mesh.add_triangle(top, color, hover)

    rings = [polygon.exterior, *polygon.interiors]
    for ring in rings:
        coordinates = list(ring.coords)
        for start, end in zip(coordinates, coordinates[1:]):
            x0, y0 = start
            x1, y1 = end
            mesh.add_triangle(
                [(x0, y0, entry_time), (x1, y1, entry_time), (x1, y1, exit_time)],
                color,
                hover,
            )
            mesh.add_triangle(
                [(x0, y0, entry_time), (x1, y1, exit_time), (x0, y0, exit_time)],
                color,
                hover,
            )


def build_orientation_cache(problem: dict[str, Any]) -> list[list[Polygon | MultiPolygon]]:
    return [
        [orientation_union(orientation) for orientation in block.get("shape", [])]
        for block in problem.get("blocks", [])
    ]


def build_snapshot_meshes(
    problem: dict[str, Any],
    snapshot: dict[str, Any],
    orientation_cache: list[list[Polygon | MultiPolygon]],
) -> list[Mesh]:
    meshes = [Mesh() for _ in problem.get("bays", [])]
    changed = {int(block_id) for block_id in snapshot.get("changed_block_ids", [])}
    selected = {int(block_id) for block_id in snapshot.get("selected_block_ids", [])}

    for scheduled in snapshot.get("schedule", []):
        block_id = int(scheduled["block_id"])
        bay_id = int(scheduled["bay_id"])
        orient_idx = int(scheduled["orient_idx"])
        geometry = orientation_cache[block_id][orient_idx]
        geometry = affinity.translate(
            geometry,
            xoff=float(scheduled["x"]),
            yoff=float(scheduled["y"]),
        )
        entry_time = float(scheduled["entry_time"])
        exit_time = float(scheduled["exit_time"])
        color = block_color(block_id)
        if block_id in selected:
            color = "rgb(118,107,143)"
        elif block_id in changed:
            color = "rgb(86,125,145)"
        hover = (
            f"B{block_id}<br>bay={bay_id}<br>orient={orient_idx}"
            f"<br>x={scheduled['x']} y={scheduled['y']}"
            f"<br>entry={scheduled['entry_time']} exit={scheduled['exit_time']}"
            f"<br>changed={block_id in changed} selected={block_id in selected}"
        )
        for polygon in polygon_parts(geometry):
            add_extruded_polygon(
                meshes[bay_id], polygon, entry_time, exit_time, color, hover
            )
    return meshes


def mesh_trace(mesh: Mesh, scene: str, name: str) -> go.Mesh3d:
    return go.Mesh3d(
        x=mesh.x,
        y=mesh.y,
        z=mesh.z,
        i=mesh.i,
        j=mesh.j,
        k=mesh.k,
        facecolor=mesh.facecolor,
        text=mesh.hovertext,
        hovertemplate="%{text}<extra></extra>",
        flatshading=True,
        opacity=0.82,
        lighting={"ambient": 0.65, "diffuse": 0.75, "specular": 0.1},
        lightposition={"x": 1000, "y": 1000, "z": 2000},
        scene=scene,
        name=name,
        showlegend=False,
    )


def snapshot_label(snapshot: dict[str, Any], index: int) -> str:
    return (
        f"#{index + 1}  {float(snapshot.get('elapsed', 0)):.2f}s  "
        f"{snapshot.get('neighbor') or '-'}  {snapshot.get('reason') or '-'}"
    )


def build_figure(
    problem: dict[str, Any], snapshots: list[dict[str, Any]]
) -> go.Figure:
    bays = problem.get("bays", [])
    bay_count = len(bays)
    subplot_titles = [
        f"Bay {bay_id}: {bay['width']} × {bay['height']}"
        for bay_id, bay in enumerate(bays)
    ]
    figure = make_subplots(
        rows=1,
        cols=bay_count,
        specs=[[{"type": "scene"} for _ in bays]],
        subplot_titles=subplot_titles,
        horizontal_spacing=min(0.04, 0.2 / max(1, bay_count)),
    )

    orientation_cache = build_orientation_cache(problem)
    all_meshes = [
        build_snapshot_meshes(problem, snapshot, orientation_cache)
        for snapshot in snapshots
    ]
    for bay_id, mesh in enumerate(all_meshes[0]):
        figure.add_trace(
            mesh_trace(mesh, f"scene{bay_id + 1}" if bay_id else "scene", f"Bay {bay_id}"),
            row=1,
            col=bay_id + 1,
        )

    frames = []
    for snapshot_index, (snapshot, meshes) in enumerate(zip(snapshots, all_meshes)):
        frames.append(
            go.Frame(
                name=str(snapshot_index),
                data=[
                    mesh_trace(
                        mesh,
                        f"scene{bay_id + 1}" if bay_id else "scene",
                        f"Bay {bay_id}",
                    )
                    for bay_id, mesh in enumerate(meshes)
                ],
                traces=list(range(bay_count)),
                layout=go.Layout(
                    title_text=f"Annealing 3D — {snapshot_label(snapshot, snapshot_index)}"
                ),
            )
        )
    figure.frames = frames

    max_width = max(float(bay.get("width", 1)) for bay in bays)
    max_height = max(float(bay.get("height", 1)) for bay in bays)
    final_schedule = snapshots[-1].get("schedule", [])
    time_max = max(
        1.0,
        *(float(scheduled.get("exit_time", 0)) for scheduled in final_schedule),
    )
    for bay_id, bay in enumerate(bays):
        scene_name = "scene" if bay_id == 0 else f"scene{bay_id + 1}"
        figure.layout[scene_name].update(
            xaxis={"title": "x", "range": [0, float(bay["width"])]},
            yaxis={"title": "y", "range": [0, float(bay["height"])]},
            zaxis={"title": "t", "range": [0, time_max]},
            aspectmode="manual",
            aspectratio={
                "x": float(bay["width"]) / max_width,
                "y": float(bay["height"]) / max_width,
                "z": 1.4,
            },
            camera={"eye": {"x": 1.5, "y": 1.5, "z": 1.1}},
        )

    steps = [
        {
            "args": [
                [str(index)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": str(index + 1),
            "method": "animate",
        }
        for index in range(len(snapshots))
    ]
    figure.update_layout(
        title=f"Annealing 3D — {snapshot_label(snapshots[0], 0)}",
        width=max(1200, 430 * bay_count),
        height=760,
        margin={"l": 20, "r": 20, "t": 90, "b": 120},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "snapshot: "},
                "pad": {"t": 45},
                "steps": steps,
            }
        ],
    )
    return figure


def main() -> int:
    args = parse_args()
    root = repo_root()
    problem_path = resolve_path(root, args.problem)
    visualize_dir = resolve_path(root, args.visualize_dir)
    worker_path = visualize_dir / f"worker_{args.worker_id}.jsonl"
    if not problem_path.is_file():
        print(f"error: problem not found: {problem_path}", file=sys.stderr)
        return 1
    if not worker_path.is_file():
        print(f"error: worker snapshot not found: {worker_path}", file=sys.stderr)
        return 1

    try:
        problem = load_json(problem_path)
        snapshots = load_snapshots(worker_path)
        if not snapshots:
            raise ValueError(f"no snapshots in {worker_path}")
        figure = build_figure(problem, snapshots)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = (
        Path(args.out)
        if args.out
        else visualize_dir / f"worker_{args.worker_id}_3d.html"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f"created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
