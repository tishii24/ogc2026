# type: ignore
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from shapely import affinity
from shapely.geometry import Polygon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Rust crane collision against Shapely and report false negatives."
    )
    parser.add_argument("--problem", required=True, help="Problem JSON path")
    parser.add_argument(
        "--checker",
        default="target/release/check_collision",
        help="Path to Rust check_collision binary. default: target/release/check_collision",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=1,
        help="Extra integer margin added around bbox-overlap dx/dy range. default: 1",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-10,
        help="Positive area threshold for Shapely hit. default: 1e-10",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many queries. 0 means no limit. default: 0",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10000,
        help="Print progress every N queries. 0 disables. default: 10000",
    )
    parser.add_argument(
        "--save-false-positive-dir",
        default="",
        help="Directory to save false positive plots and json metadata. default: disabled",
    )
    parser.add_argument(
        "--max-save-false-positive",
        type=int,
        default=20,
        help="Maximum number of false positives to save. default: 20",
    )
    parser.add_argument(
        "--false-positive-format",
        choices=("png", "svg"),
        default="png",
        help="Plot format for false positives. default: png",
    )
    return parser.parse_args()


def load_problem(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def layer_bbox(layer: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in layer]
    ys = [p[1] for p in layer]
    return min(xs), min(ys), max(xs), max(ys)


def merge_bbox(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def build_geoms(problem: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    geoms: dict[tuple[int, int], dict[str, Any]] = {}
    for block_id, block in enumerate(problem["blocks"]):
        for orient_idx, orientation in enumerate(block["shape"]):
            polygons = []
            bbox = None
            for layer in orientation["layers"]:
                polygon = Polygon(layer)
                polygons.append(polygon)
                bbox = (
                    layer_bbox(layer)
                    if bbox is None
                    else merge_bbox(bbox, layer_bbox(layer))
                )
            if bbox is None:
                bbox = (0.0, 0.0, 0.0, 0.0)
            geoms[(block_id, orient_idx)] = {
                "polygons": polygons,
                "bbox": bbox,
            }
    return geoms


def delta_range(
    moving_bbox: tuple[float, float, float, float],
    fixed_bbox: tuple[float, float, float, float],
    margin: int,
) -> tuple[int, int, int, int]:
    return (
        math.floor(moving_bbox[0] - fixed_bbox[2]) - margin,
        math.ceil(moving_bbox[2] - fixed_bbox[0]) + margin,
        math.floor(moving_bbox[1] - fixed_bbox[3]) - margin,
        math.ceil(moving_bbox[3] - fixed_bbox[1]) + margin,
    )


def bbox_may_overlap(
    moving_bbox: tuple[float, float, float, float],
    fixed_bbox: tuple[float, float, float, float],
    dx: int,
    dy: int,
) -> bool:
    return not (
        moving_bbox[2] < fixed_bbox[0] + dx
        or fixed_bbox[2] + dx < moving_bbox[0]
        or moving_bbox[3] < fixed_bbox[1] + dy
        or fixed_bbox[3] + dy < moving_bbox[1]
    )


def crane_hit_shapely(
    geoms: dict[tuple[int, int], dict[str, Any]],
    query: dict[str, Any],
    eps: float,
) -> bool:
    moving = query["moving"]
    fixed = query["fixed"]
    moving_geom = geoms[(moving["block_id"], moving["orient_idx"])]
    fixed_geom = geoms[(fixed["block_id"], fixed["orient_idx"])]
    dx = fixed["x"] - moving["x"]
    dy = fixed["y"] - moving["y"]

    for k, moving_poly in enumerate(moving_geom["polygons"]):
        for j in range(k, len(fixed_geom["polygons"])):
            fixed_poly = fixed_geom["polygons"][j]
            if not bbox_may_overlap(
                moving_poly.bounds,
                fixed_poly.bounds,
                dx,
                dy,
            ):
                continue
            shifted_fixed = affinity.translate(fixed_poly, xoff=dx, yoff=dy)
            if moving_poly.intersection(shifted_fixed).area > eps:
                return True
    return False


def layer_pair_details(
    geoms: dict[tuple[int, int], dict[str, Any]], query: dict[str, Any], limit: int = 12
) -> list[dict[str, Any]]:
    moving = query["moving"]
    fixed = query["fixed"]
    moving_geom = geoms[(moving["block_id"], moving["orient_idx"])]
    fixed_geom = geoms[(fixed["block_id"], fixed["orient_idx"])]
    dx = fixed["x"] - moving["x"]
    dy = fixed["y"] - moving["y"]
    details = []

    for k, moving_poly in enumerate(moving_geom["polygons"]):
        for j in range(k, len(fixed_geom["polygons"])):
            shifted_fixed = affinity.translate(
                fixed_geom["polygons"][j], xoff=dx, yoff=dy
            )
            details.append(
                {
                    "moving_layer": k,
                    "fixed_layer": j,
                    "distance": moving_poly.distance(shifted_fixed),
                    "intersection_area": moving_poly.intersection(shifted_fixed).area,
                    "touches": moving_poly.touches(shifted_fixed),
                }
            )

    details.sort(key=lambda item: (item["distance"], -item["intersection_area"]))
    return details[:limit]


def draw_polygon(
    ax: Any, poly: Polygon, *, color: str, alpha: float, label: str
) -> None:
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5)
    centroid = poly.centroid
    ax.text(centroid.x, centroid.y, label, color=color, fontsize=8)


def plot_false_positive(
    geoms: dict[tuple[int, int], dict[str, Any]],
    query: dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    moving = query["moving"]
    fixed = query["fixed"]
    dx = fixed["x"] - moving["x"]
    dy = fixed["y"] - moving["y"]
    moving_geom = geoms[(moving["block_id"], moving["orient_idx"])]
    fixed_geom = geoms[(fixed["block_id"], fixed["orient_idx"])]

    fig, ax = plt.subplots(figsize=(7, 7))
    for k, poly in enumerate(moving_geom["polygons"]):
        draw_polygon(ax, poly, color="tab:blue", alpha=0.25, label=f"M{k}")
    for j, poly in enumerate(fixed_geom["polygons"]):
        shifted = affinity.translate(poly, xoff=dx, yoff=dy)
        draw_polygon(ax, shifted, color="tab:red", alpha=0.25, label=f"F{j}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"False positive: M={moving['block_id']}:{moving['orient_idx']} "
        f"F={fixed['block_id']}:{fixed['orient_idx']} dx={dx} dy={dy}"
    )
    ax.grid(True, linewidth=0.3)
    ax.autoscale()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_false_positive(
    save_dir: Path,
    index: int,
    problem: dict[str, Any],
    problem_path: str,
    geoms: dict[tuple[int, int], dict[str, Any]],
    query: dict[str, Any],
    image_format: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fp_{index:06d}"
    moving = query["moving"]
    fixed = query["fixed"]
    dx = fixed["x"] - moving["x"]
    dy = fixed["y"] - moving["y"]
    metadata = {
        "problem": problem.get("name", problem_path),
        "query": query,
        "dx": dx,
        "dy": dy,
        "layer_pairs": layer_pair_details(geoms, query),
    }
    (save_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_false_positive(geoms, query, save_dir / f"{stem}.{image_format}")


def start_checker(checker: str, problem: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [checker, problem],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def ask_rust(proc: subprocess.Popen[str], query: dict[str, Any]) -> bool:
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("checker stdin/stdout is not available")
    proc.stdin.write(json.dumps(query, separators=(",", ":")) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        raise RuntimeError(f"checker terminated unexpectedly\nSTDERR:\n{stderr}")
    result = json.loads(line)
    if result["id"] != query["id"]:
        raise RuntimeError(
            f"result id mismatch: query={query['id']} result={result['id']}"
        )
    return bool(result["hit"])


def iter_queries(
    problem: dict[str, Any], geoms: dict[tuple[int, int], dict[str, Any]], margin: int
):
    query_id = 0
    blocks = problem["blocks"]
    for moving_id, moving_block in enumerate(blocks):
        for fixed_id, fixed_block in enumerate(blocks):
            if moving_id == fixed_id:
                continue
            for moving_o in range(len(moving_block["shape"])):
                for fixed_o in range(len(fixed_block["shape"])):
                    moving_bbox = geoms[(moving_id, moving_o)]["bbox"]
                    fixed_bbox = geoms[(fixed_id, fixed_o)]["bbox"]
                    min_dx, max_dx, min_dy, max_dy = delta_range(
                        moving_bbox, fixed_bbox, margin
                    )
                    for dx in range(min_dx, max_dx + 1):
                        for dy in range(min_dy, max_dy + 1):
                            yield {
                                "id": query_id,
                                "moving": {
                                    "block_id": moving_id,
                                    "orient_idx": moving_o,
                                    "x": 0,
                                    "y": 0,
                                },
                                "fixed": {
                                    "block_id": fixed_id,
                                    "orient_idx": fixed_o,
                                    "x": dx,
                                    "y": dy,
                                },
                            }
                            query_id += 1


def main() -> int:
    args = parse_args()
    problem_path = str(Path(args.problem))
    checker_path = str(Path(args.checker))
    problem = load_problem(problem_path)
    geoms = build_geoms(problem)
    proc = start_checker(checker_path, problem_path)

    tested = 0
    python_hit = 0
    rust_hit = 0
    false_positive = 0
    saved_false_positive = 0
    save_false_positive_dir = (
        Path(args.save_false_positive_dir) if args.save_false_positive_dir else None
    )

    try:
        for query in iter_queries(problem, geoms, args.margin):
            rust = ask_rust(proc, query)
            py = crane_hit_shapely(geoms, query, args.eps)

            tested += 1
            if rust:
                rust_hit += 1
            if py:
                python_hit += 1
            if rust and not py:
                false_positive += 1
                if (
                    save_false_positive_dir is not None
                    and saved_false_positive < args.max_save_false_positive
                ):
                    saved_false_positive += 1
                    save_false_positive(
                        save_false_positive_dir,
                        saved_false_positive,
                        problem,
                        problem_path,
                        geoms,
                        query,
                        args.false_positive_format,
                    )
            if py and not rust:
                print("FALSE NEGATIVE")
                print(f"problem={problem.get('name', problem_path)}")
                print(f"tested={tested}")
                print(json.dumps(query, indent=2))
                return 1

            if args.progress_interval and tested % args.progress_interval == 0:
                print(
                    f"tested={tested} python_hit={python_hit} rust_hit={rust_hit} "
                    f"false_positive={false_positive} saved_false_positive={saved_false_positive}",
                    file=sys.stderr,
                )
            if args.limit and tested >= args.limit:
                break
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(
        f"tested={tested} python_hit={python_hit} rust_hit={rust_hit} "
        f"false_positive={false_positive} saved_false_positive={saved_false_positive} "
        f"false_negative=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
