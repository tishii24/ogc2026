# ruff: noqa
# type: ignore

#!/usr/bin/env python3
"""Run a composed OGC 2026 solution and append scores to log/score.csv."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Iterator

VERSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
CSV_COLUMNS = [
    "timestamp",
    "version",
    "case",
    "timelimit",
    "elapsed",
    "feasible",
    "stage",
    "objective",
    "obj1",
    "obj2",
    "obj3",
    "n_blocks",
    "error",
]

MAX_ALGORITHM_THREADS = 4
THREAD_LIMIT_ENV_KEYS = [
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run solutions/{version}/myalgorithm.py on local cases."
    )
    parser.add_argument("version", help="Version directory under solutions/")
    parser.add_argument("--case", help="Problem JSON path for a single case.")
    parser.add_argument(
        "--params",
        help="YAML parameter file. default: solutions/{version}/params.yaml",
    )
    parser.add_argument(
        "--suite",
        help='Suite JSON path. Format: {"cases": ["train/prob_1.json", ...]}. default: train/*.json',
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=60.0,
        help="Timelimit passed to algorithm(). default: 60",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Enable local Rust logs and panic on collision fallback.",
    )

    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_version(version: str) -> None:
    if not VERSION_RE.fullmatch(version):
        raise ValueError(
            "version must contain only letters, digits, underscore, dot, or hyphen"
        )


def apply_thread_limit(max_threads: int = MAX_ALGORITHM_THREADS) -> None:
    value = str(max_threads)
    for key in THREAD_LIMIT_ENV_KEYS:
        os.environ[key] = value


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))
    ]


def load_suite_cases(root: Path, suite: str) -> list[str]:
    path = Path(suite)
    if not path.is_absolute():
        path = root / path
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("suite must be a JSON object")
    cases = data.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, str) for case in cases):
        raise ValueError("suite.cases must be a list of strings")
    return cases


def collect_cases(root: Path, args: argparse.Namespace) -> list[Path]:
    if args.case and args.suite:
        raise ValueError("--case and --suite cannot be used together")
    if args.case:
        path = Path(args.case)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"case not found: {args.case}")
        return [path]

    patterns = load_suite_cases(root, args.suite) if args.suite else ["train/*.json"]

    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = (
            glob.glob(str(root / pattern))
            if not Path(pattern).is_absolute()
            else glob.glob(pattern)
        )
        matches.sort(key=lambda path: natural_key(Path(path)))
        if not matches:
            path = root / pattern if not Path(pattern).is_absolute() else Path(pattern)
            matches = [str(path)]
        for match in matches:
            path = Path(match).resolve()
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)

    return paths


def load_myalgorithm(myalgorithm_path: Path):
    module_name = f"myalgorithm_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, myalgorithm_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module spec: {myalgorithm_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "algorithm"):
        raise AttributeError(f"{myalgorithm_path} does not define algorithm()")
    return module


def load_checker(root: Path):
    sys.path.insert(0, str(root / "ogc2026" / "alg_tester"))
    from utils import check_feasibility

    return check_feasibility


def timelimit_dir_name(timelimit: float) -> str:
    return f"{timelimit:g}"


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value or "unknown"


def testcase_name(prob_info: dict[str, Any] | None, case_path: Path) -> str:
    if prob_info is not None:
        name = prob_info.get("name")
        if isinstance(name, str) and name:
            return safe_component(name)
    return safe_component(case_path.stem)


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_run_artifact_dir(
    root: Path, version: str, timelimit: float, testcase: str
) -> Path:
    output_dir = root / "log" / version / timelimit_dir_name(timelimit) / testcase
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


@contextmanager
def tee_stderr(path: Path) -> Iterator[None]:
    saved_stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    log_file = path.open("wb", buffering=0)
    pump_error: list[BaseException] = []

    def pump() -> None:
        try:
            while chunk := os.read(read_fd, 65536):
                remaining = memoryview(chunk)
                while remaining:
                    remaining = remaining[os.write(saved_stderr_fd, remaining) :]
                log_file.write(chunk)
        except BaseException as exc:
            pump_error.append(exc)
        finally:
            os.close(read_fd)
            log_file.close()

    thread = Thread(target=pump, daemon=True)
    thread.start()
    try:
        sys.stderr.flush()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stderr_fd, 2)
        thread.join()
        os.close(saved_stderr_fd)
        if pump_error:
            print(f"warning: failed to tee stderr: {pump_error[0]}", file=sys.stderr)


def save_run_artifacts(
    output_dir: Path,
    params_path: Path,
    meta: dict[str, Any],
    solution: Any | None,
    result: dict[str, Any] | None,
    error: str,
) -> None:
    (output_dir / "stderr.log").touch(exist_ok=True)
    shutil.copy2(params_path, output_dir / "params.yaml")
    write_json(output_dir / "meta.json", meta)
    if solution is not None:
        write_json(output_dir / "solution.json", solution)
    if result is not None:
        write_json(output_dir / "result.json", result)
    if error:
        (output_dir / "error.txt").write_text(error + "\n", encoding="utf-8")


def run_case(
    root: Path,
    version: str,
    myalgorithm_path: Path,
    check_feasibility,
    case_path: Path,
    timelimit: float,
    params_path: Path,
) -> dict[str, Any]:
    apply_thread_limit()
    params_path = params_path.resolve()
    os.environ["OGC_PARAMS_PATH"] = str(params_path)
    params_sha256 = hashlib.sha256(params_path.read_bytes()).hexdigest()
    timestamp = datetime.now().isoformat(timespec="seconds")
    rel_case = (
        str(case_path.relative_to(root))
        if case_path.is_relative_to(root)
        else str(case_path)
    )

    row: dict[str, Any] = {
        "timestamp": timestamp,
        "version": version,
        "case": rel_case,
        "timelimit": timelimit,
        "elapsed": "",
        "feasible": False,
        "stage": 0,
        "objective": "",
        "obj1": "",
        "obj2": "",
        "obj3": "",
        "n_blocks": "",
        "error": "",
    }

    prob_info: dict[str, Any] | None = None
    solution: Any | None = None
    result: dict[str, Any] | None = None
    testcase = testcase_name(None, case_path)
    output_dir: Path | None = None

    started = time.perf_counter()
    try:
        with case_path.open(encoding="utf-8") as f:
            prob_info = json.load(f)
        testcase = testcase_name(prob_info, case_path)
        row["n_blocks"] = len(prob_info.get("blocks", []))
        output_dir = prepare_run_artifact_dir(root, version, timelimit, testcase)

        with tee_stderr(output_dir / "stderr.log"):
            module = load_myalgorithm(myalgorithm_path)
            solution = module.algorithm(prob_info, timelimit)
            result = check_feasibility(prob_info, solution)
        elapsed = time.perf_counter() - started

        row.update(
            {
                "elapsed": f"{elapsed:.6f}",
                "feasible": bool(result.get("feasible")),
                "stage": result.get("stage", 0),
                "objective": result.get("objective")
                if result.get("objective") is not None
                else "",
                "obj1": result.get("obj1") if result.get("obj1") is not None else "",
                "obj2": result.get("obj2") if result.get("obj2") is not None else "",
                "obj3": result.get("obj3") if result.get("obj3") is not None else "",
            }
        )
        if not result.get("feasible"):
            row["error"] = "; ".join(result.get("violations", [])[:3])
    except Exception as exc:
        elapsed = time.perf_counter() - started
        row["elapsed"] = f"{elapsed:.6f}"
        row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()

    meta = {
        "timestamp": timestamp,
        "version": version,
        "case": rel_case,
        "testcase": testcase,
        "timelimit": timelimit,
        "params_path": str(params_path),
        "params_sha256": params_sha256,
        "elapsed": float(row["elapsed"]) if row["elapsed"] else None,
        "feasible": bool(row["feasible"]),
        "stage": row["stage"],
        "objective": row["objective"] if row["objective"] != "" else None,
        "error": row["error"],
    }
    if output_dir is None:
        output_dir = prepare_run_artifact_dir(root, version, timelimit, testcase)
    save_run_artifacts(
        output_dir=output_dir,
        params_path=params_path,
        meta=meta,
        solution=solution,
        result=result,
        error=row["error"],
    )

    return row


def append_score(root: Path, row: dict[str, Any]) -> None:
    log_dir = root / "log"
    log_dir.mkdir(exist_ok=True)
    score_path = log_dir / "score.csv"
    write_header = not score_path.exists()

    with score_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)




def print_row(index: int, total: int, row: dict[str, Any]) -> None:
    status = "OK" if row["feasible"] else "NG"
    objective = row["objective"] if row["objective"] != "" else "-"
    elapsed = f"{float(row['elapsed']):.4f}s" if row["elapsed"] else "-"
    print(
        f"[{index:2}/{total}] {status} {row['case']:24s} "
        f"obj={objective:16} elapsed={elapsed}"
    )
    if row["error"]:
        print(f"  error: {row['error']}")


def main() -> int:
    args = parse_args()
    root = repo_root()
    if args.local:
        os.environ["OGC_LOCAL"] = "1"
    else:
        os.environ.pop("OGC_LOCAL", None)

    try:
        validate_version(args.version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    solution_dir = root / "solutions" / args.version
    myalgorithm_path = solution_dir / "myalgorithm.py"
    if not myalgorithm_path.is_file():
        print(
            f"error: {myalgorithm_path} not found. Run tools/composer.py first.",
            file=sys.stderr,
        )
        return 1

    params_path = Path(args.params) if args.params else solution_dir / "params.yaml"
    if not params_path.is_absolute():
        params_path = root / params_path
    params_path = params_path.resolve()
    if not params_path.is_file():
        print(f"error: params file not found: {params_path}", file=sys.stderr)
        return 1

    try:
        cases = collect_cases(root, args)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not cases:
        print("error: no cases found", file=sys.stderr)
        return 1

    feasible_count = 0
    check_feasibility = load_checker(root)
    for index, case_path in enumerate(cases, start=1):
        row = run_case(
            root,
            args.version,
            myalgorithm_path,
            check_feasibility,
            case_path,
            args.timelimit,
            params_path,
        )
        append_score(root, row)
        if row["feasible"]:
            feasible_count += 1
        print_row(index, len(cases), row)
        if not row["feasible"]:
            print(f"summary: feasible {feasible_count}/{index}")
            print("log: log/score.csv")
            return 1

    print(f"summary: feasible {feasible_count}/{len(cases)}")
    print("log: log/score.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
