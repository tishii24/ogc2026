#!/usr/bin/env python3
"""Run a composed OGC 2026 solution and append scores to log/score.csv."""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run solutions/{version}/myalgorithm.py on local cases."
    )
    parser.add_argument("version", help="Version directory under solutions/")
    parser.add_argument("--case", help="Problem JSON path for a single case.")
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
        "--jobs",
        type=int,
        default=1,
        help="Number of cases to run in parallel. default: 1",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_version(version: str) -> None:
    if not VERSION_RE.fullmatch(version):
        raise ValueError(
            "version must contain only letters, digits, underscore, dot, or hyphen"
        )


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


def save_run_artifacts(
    root: Path,
    version: str,
    timelimit: float,
    testcase: str,
    meta: dict[str, Any],
    solution: Any | None,
    result: dict[str, Any] | None,
    error: str,
) -> Path:
    output_dir = root / "log" / version / timelimit_dir_name(timelimit) / testcase
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    write_json(output_dir / "meta.json", meta)
    if solution is not None:
        write_json(output_dir / "solution.json", solution)
    if result is not None:
        write_json(output_dir / "result.json", result)
    if error:
        (output_dir / "error.txt").write_text(error + "\n", encoding="utf-8")
    return output_dir


def run_case(
    root: Path,
    version: str,
    myalgorithm_path: Path,
    check_feasibility,
    case_path: Path,
    timelimit: float,
) -> dict[str, Any]:
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

    started = time.perf_counter()
    try:
        with case_path.open(encoding="utf-8") as f:
            prob_info = json.load(f)
        testcase = testcase_name(prob_info, case_path)
        row["n_blocks"] = len(prob_info.get("blocks", []))

        module = load_myalgorithm(myalgorithm_path)
        solution = module.algorithm(prob_info, timelimit)
        elapsed = time.perf_counter() - started

        result = check_feasibility(prob_info, solution)
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
        "elapsed": float(row["elapsed"]) if row["elapsed"] else None,
        "feasible": bool(row["feasible"]),
        "stage": row["stage"],
        "objective": row["objective"] if row["objective"] != "" else None,
        "error": row["error"],
    }
    save_run_artifacts(
        root=root,
        version=version,
        timelimit=timelimit,
        testcase=testcase,
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


def run_case_worker(args: tuple[str, str, str, str, float]) -> dict[str, Any]:
    root_s, version, myalgorithm_path_s, case_path_s, timelimit = args
    root = Path(root_s)
    check_feasibility = load_checker(root)
    return run_case(
        root=root,
        version=version,
        myalgorithm_path=Path(myalgorithm_path_s),
        check_feasibility=check_feasibility,
        case_path=Path(case_path_s),
        timelimit=timelimit,
    )


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

    try:
        validate_version(args.version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    myalgorithm_path = root / "solutions" / args.version / "myalgorithm.py"
    if not myalgorithm_path.is_file():
        print(
            f"error: {myalgorithm_path} not found. Run tools/composer.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        cases = collect_cases(root, args)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not cases:
        print("error: no cases found", file=sys.stderr)
        return 1

    if args.jobs < 1:
        print("error: --jobs must be >= 1", file=sys.stderr)
        return 1

    feasible_count = 0
    if args.jobs == 1:
        check_feasibility = load_checker(root)
        for index, case_path in enumerate(cases, start=1):
            row = run_case(
                root,
                args.version,
                myalgorithm_path,
                check_feasibility,
                case_path,
                args.timelimit,
            )
            append_score(root, row)
            if row["feasible"]:
                feasible_count += 1
            print_row(index, len(cases), row)
    else:
        worker_args = [
            (
                str(root),
                args.version,
                str(myalgorithm_path),
                str(case_path),
                args.timelimit,
            )
            for case_path in cases
        ]
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_index = {
                executor.submit(run_case_worker, arg): index
                for index, arg in enumerate(worker_args, start=1)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    row = future.result()
                except Exception as exc:
                    case_path = cases[index - 1]
                    rel_case = (
                        str(case_path.relative_to(root))
                        if case_path.is_relative_to(root)
                        else str(case_path)
                    )
                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "version": args.version,
                        "case": rel_case,
                        "timelimit": args.timelimit,
                        "elapsed": "",
                        "feasible": False,
                        "stage": 0,
                        "objective": "",
                        "obj1": "",
                        "obj2": "",
                        "obj3": "",
                        "n_blocks": "",
                        "error": "".join(
                            traceback.format_exception_only(type(exc), exc)
                        ).strip(),
                    }
                append_score(root, row)
                if row["feasible"]:
                    feasible_count += 1
                print_row(index, len(cases), row)

    print(f"summary: feasible {feasible_count}/{len(cases)}")
    print("log: log/score.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
