# ruff: noqa
# type: ignore

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

from shapely.geometry import Polygon

VALIDATION_RESERVE_SECONDS = 1.0
RETURN_BUFFER_SECONDS = 0.5


class FeasibilityTimeout(Exception):
    pass


def _log(started: float, message: str) -> None:
    print(
        f"[{time.perf_counter() - started:.4f}] [myalgorithm] {message}",
        file=sys.stderr,
    )


def _check_time(stop_time: float) -> None:
    if time.perf_counter() >= stop_time:
        raise FeasibilityTimeout


def _make_polygon(vertices: list[list[float]]):
    if len(vertices) < 3:
        return None
    try:
        polygon = Polygon(vertices)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return None if polygon.is_empty else polygon
    except Exception:
        return None


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def check_feasibility_fast(
    prob_info: dict[str, Any], solution: dict[str, Any], stop_time: float
) -> bool | None:
    try:
        _check_time(stop_time)
        blocks_data = prob_info["blocks"]
        bays_data = prob_info["bays"]
        n_blocks = len(blocks_data)
        n_bays = len(bays_data)
        raw_operations = solution.get("operations")
        if not isinstance(raw_operations, dict) or n_bays == 0:
            return False

        operations: list[tuple[int, list[dict[str, Any]]]] = []
        assignments: list[dict[str, Any] | None] = [None] * n_blocks
        entry_counts = [0] * n_blocks
        exit_counts = [0] * n_blocks

        for time_key, ops in raw_operations.items():
            _check_time(stop_time)
            try:
                operation_time = int(time_key)
            except (TypeError, ValueError):
                return False
            if not isinstance(ops, list):
                return False
            operations.append((operation_time, ops))
            for op in ops:
                if not isinstance(op, dict) or any(
                    key not in op for key in ("type", "block_id", "bay_id")
                ):
                    return False
                block_id = op["block_id"]
                if not isinstance(block_id, int) or not 0 <= block_id < n_blocks:
                    return False
                if op["type"] == "ENTRY":
                    entry_counts[block_id] += 1
                    assignments[block_id] = {
                        "block_id": block_id,
                        "bay_id": op["bay_id"],
                        "x": op.get("x", 0.0),
                        "y": op.get("y", 0.0),
                        "orient_idx": op.get("orient_idx", 0),
                        "entry_time": operation_time,
                        "exit_time": None,
                    }
                elif op["type"] == "EXIT":
                    exit_counts[block_id] += 1
                    if assignments[block_id] is not None:
                        assignments[block_id]["exit_time"] = operation_time
                else:
                    return False

        if any(count != 1 for count in entry_counts) or any(
            count != 1 for count in exit_counts
        ):
            return False

        placed: list[dict[str, Any]] = []
        for block_id, assignment in enumerate(assignments):
            _check_time(stop_time)
            if assignment is None or assignment["exit_time"] is None:
                return False
            bay_id = assignment["bay_id"]
            orient_idx = assignment["orient_idx"]
            if not isinstance(bay_id, int) or not 0 <= bay_id < n_bays:
                return False
            if not isinstance(orient_idx, int) or not 0 <= orient_idx < len(
                blocks_data[block_id]["shape"]
            ):
                return False
            if (
                assignment["exit_time"] - assignment["entry_time"]
                < blocks_data[block_id]["processing_time"] - 1e-6
                or assignment["entry_time"]
                < blocks_data[block_id]["release_time"] - 1e-6
            ):
                return False

            x = int(round(assignment["x"]))
            y = int(round(assignment["y"]))
            raw_layers = blocks_data[block_id]["shape"][orient_idx]["layers"]
            ref_x, ref_y = (raw_layers[0][0] if raw_layers and raw_layers[0] else (0.0, 0.0))
            layers = []
            all_vertices = []
            for raw_layer in raw_layers:
                vertices = [
                    [vx + x - ref_x, vy + y - ref_y] for vx, vy in raw_layer
                ]
                all_vertices.extend(vertices)
                layers.append(_make_polygon(vertices))
            if all_vertices:
                xs = [vertex[0] for vertex in all_vertices]
                ys = [vertex[1] for vertex in all_vertices]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                bbox = (float(x), float(y), float(x + 1), float(y + 1))
            placed.append(
                {
                    "bay_id": bay_id,
                    "layers": layers,
                    "bbox": bbox,
                }
            )

        obstruction_cache: dict[tuple[int, int], bool] = {}

        def obstructed(moving_id: int, fixed_id: int) -> bool:
            key = (moving_id, fixed_id)
            cached = obstruction_cache.get(key)
            if cached is not None:
                return cached
            _check_time(stop_time)
            moving = placed[moving_id]
            fixed = placed[fixed_id]
            if not _bbox_overlap(moving["bbox"], fixed["bbox"]):
                obstruction_cache[key] = False
                return False
            for moving_layer, moving_polygon in enumerate(moving["layers"]):
                if moving_polygon is None:
                    continue
                for fixed_layer in range(moving_layer, len(fixed["layers"])):
                    _check_time(stop_time)
                    fixed_polygon = fixed["layers"][fixed_layer]
                    if fixed_polygon is None:
                        continue
                    try:
                        intersection = moving_polygon.intersection(fixed_polygon)
                    except Exception:
                        continue
                    if not intersection.is_empty and intersection.area > 0:
                        obstruction_cache[key] = True
                        return True
            obstruction_cache[key] = False
            return False

        operations.sort(key=lambda item: item[0])
        present: list[set[int]] = [set() for _ in range(n_bays)]
        for _, ops in operations:
            _check_time(stop_time)
            seen_entry = False
            for op in ops:
                if op["type"] == "EXIT" and seen_entry:
                    return False
                if op["type"] == "ENTRY":
                    seen_entry = True

            for op in ops:
                _check_time(stop_time)
                block_id = op["block_id"]
                bay_id = op["bay_id"]
                assignment = assignments[block_id]
                if assignment is None or assignment["bay_id"] != bay_id:
                    return False
                if op["type"] == "ENTRY":
                    bbox = placed[block_id]["bbox"]
                    bay = bays_data[bay_id]
                    if not (
                        bbox[0] >= 0
                        and bbox[1] >= 0
                        and bbox[2] <= bay["width"]
                        and bbox[3] <= bay["height"]
                    ):
                        return False
                    for fixed_id in present[bay_id]:
                        if obstructed(block_id, fixed_id):
                            return False
                    present[bay_id].add(block_id)
                else:
                    if block_id not in present[bay_id]:
                        return False
                    for fixed_id in present[bay_id]:
                        if fixed_id != block_id and obstructed(block_id, fixed_id):
                            return False
                    present[bay_id].remove(block_id)

        return not any(present)
    except FeasibilityTimeout:
        return None


def _solver_path() -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parent
    name = "solver.exe" if sys.platform == "win32" else "solver"
    return root / name


def _params_path() -> pathlib.Path:
    path = os.environ.get("OGC_PARAMS_PATH")
    if path:
        return pathlib.Path(path).resolve()
    return pathlib.Path(__file__).resolve().parent / "params.yaml"


def _load_candidates(output: str) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    received_count = 0
    for line in output.splitlines():
        try:
            candidate = json.loads(line)
            score = float(candidate["score"])
            solution = candidate["solution"]
            if not math.isfinite(score) or not isinstance(solution, dict):
                continue
            key = json.dumps(solution, sort_keys=True, separators=(",", ":"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        received_count += 1
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "score": score,
                "solution": solution,
                "source_index": received_count,
            }
        )
    candidates.sort(key=lambda candidate: candidate["score"])
    return candidates, received_count


def algorithm(prob_info, timelimit=60):
    started = time.perf_counter()
    deadline = started + float(timelimit)
    solver_timelimit = max(
        0.1,
        float(timelimit) - VALIDATION_RESERVE_SECONDS - RETURN_BUFFER_SECONDS,
    )

    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("BLIS_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

    solver = _solver_path()
    try:
        solver.chmod(solver.stat().st_mode | 0o755)
    except OSError:
        pass

    _log(started, f"solver start: timelimit={solver_timelimit:.3f}s")
    process = subprocess.Popen(
        [
            str(solver),
            "--input",
            "-",
            "--timelimit",
            str(solver_timelimit),
            "--params",
            str(_params_path()),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=str(solver.parent),
    )
    try:
        output, _ = process.communicate(
            json.dumps(prob_info),
            timeout=max(0.1, deadline - time.perf_counter() - RETURN_BUFFER_SECONDS),
        )
    except subprocess.TimeoutExpired:
        _log(started, "solver timeout: killing process")
        process.kill()
        output, _ = process.communicate()

    _log(started, f"solver finished: code={process.returncode}")

    candidates, received_count = _load_candidates(output)
    _log(
        started,
        f"candidates: received={received_count}, unique={len(candidates)}",
    )
    if not candidates:
        raise RuntimeError(f"solver produced no solution candidates (code={process.returncode})")
    for rank, candidate in enumerate(candidates, start=1):
        _log(
            started,
            f"candidate: rank={rank}/{len(candidates)}, "
            f"source_index={candidate['source_index']}/{received_count}, "
            f"score={candidate['score']:.3f}",
        )

    fallback = candidates[0]
    stop_time = deadline - RETURN_BUFFER_SECONDS

    try:
        for rank, candidate in enumerate(candidates, start=1):
            if time.perf_counter() >= stop_time:
                _log(
                    started,
                    f"return: rank=1/{len(candidates)}, "
                    f"source_index={fallback['source_index']}/{received_count}, "
                    f"score={fallback['score']:.3f}, reason=deadline",
                )
                return fallback["solution"]
            validation_started = time.perf_counter()
            _log(
                started,
                f"validation start: rank={rank}/{len(candidates)}, "
                f"source_index={candidate['source_index']}/{received_count}, "
                f"score={candidate['score']:.3f}",
            )
            feasible = check_feasibility_fast(
                prob_info, candidate["solution"], stop_time
            )
            duration = time.perf_counter() - validation_started
            if feasible is None:
                _log(
                    started,
                    f"validation result: rank={rank}/{len(candidates)}, "
                    f"score={candidate['score']:.3f}, feasible=timeout, "
                    f"duration={duration:.4f}s",
                )
                _log(
                    started,
                    f"return: rank=1/{len(candidates)}, "
                    f"source_index={fallback['source_index']}/{received_count}, "
                    f"score={fallback['score']:.3f}, reason=validation-timeout",
                )
                return fallback["solution"]
            _log(
                started,
                f"validation result: rank={rank}/{len(candidates)}, "
                f"score={candidate['score']:.3f}, feasible={str(feasible).lower()}, "
                f"duration={duration:.4f}s",
            )
            if feasible:
                _log(
                    started,
                    f"return: rank={rank}/{len(candidates)}, "
                    f"source_index={candidate['source_index']}/{received_count}, "
                    f"score={candidate['score']:.3f}, reason=feasible",
                )
                return candidate["solution"]
    except Exception as exc:
        _log(started, f"candidate validation failed: {exc}")
        _log(
            started,
            f"return: rank=1/{len(candidates)}, "
            f"source_index={fallback['source_index']}/{received_count}, "
            f"score={fallback['score']:.3f}, reason=validation-error",
        )
        return fallback["solution"]

    _log(
        started,
        f"return: rank=1/{len(candidates)}, "
        f"source_index={fallback['source_index']}/{received_count}, "
        f"score={fallback['score']:.3f}, reason=all-infeasible",
    )
    return fallback["solution"]
