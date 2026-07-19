#!/usr/bin/env python3
"""Tune solver parameters step by step using runner.py and stats.py."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune parameter combinations defined in a YAML file."
    )
    parser.add_argument("config", help="Tuning YAML path.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not component:
        raise ValueError("name must contain an alphanumeric character")
    return component


def set_dotted_value(params: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current: dict[str, Any] = params
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"parameter path not found: {path}")
        current = child
    if keys[-1] not in current:
        raise ValueError(f"parameter path not found: {path}")
    current[keys[-1]] = value


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parameter_combinations(
    parameters: dict[str, list[Any]],
) -> Iterator[dict[str, Any]]:
    names = list(parameters)
    for values in itertools.product(*(parameters[name] for name in names)):
        yield dict(zip(names, values))


def step_combinations(step: dict[str, Any], step_index: int) -> tuple[str, list[dict[str, Any]]]:
    mode = str(step.get("mode", "product"))
    parameters = step.get("parameters")
    if mode == "product":
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(f"step {step_index} parameters must be a non-empty mapping")
        if not all(
            isinstance(path, str) and isinstance(values, list) and values
            for path, values in parameters.items()
        ):
            raise ValueError(
                f"step {step_index} parameter values must be non-empty lists"
            )
        return mode, list(parameter_combinations(parameters))
    if mode == "combinations":
        if not isinstance(parameters, list) or not parameters:
            raise ValueError(f"step {step_index} parameters must be a non-empty list")
        if not all(
            isinstance(candidate, dict)
            and candidate
            and all(isinstance(path, str) for path in candidate)
            for candidate in parameters
        ):
            raise ValueError(
                f"step {step_index} parameter combinations must be non-empty mappings"
            )
        return mode, parameters
    raise ValueError(f"step {step_index} has unknown mode: {mode}")


def compose_solution(root: Path, version: str, params_path: Path) -> None:
    command = [
        sys.executable,
        str(root / "tools" / "composer.py"),
        version,
        "--params",
        str(params_path),
    ]
    subprocess.run(command, cwd=root, check=True)


def run_candidate(
    root: Path,
    candidate_version: str,
    params_path: Path,
    suite: Path,
    timelimit: float,
    jobs: int,
) -> None:
    command = [
        sys.executable,
        str(root / "tools" / "runner.py"),
        candidate_version,
        "--suite",
        str(suite),
        "--timelimit",
        str(timelimit),
        "--jobs",
        str(jobs),
    ]
    compose_solution(root, candidate_version, params_path)
    try:
        subprocess.run(command, cwd=root, check=True)
    finally:
        shutil.rmtree(root / "solutions" / candidate_version, ignore_errors=True)


def load_relative_scores(
    root: Path, suite: Path, timelimit: float, versions: list[str]
) -> dict[str, float]:
    command = [
        sys.executable,
        str(root / "tools" / "stats.py"),
        "--json",
        "--include-tune",
        "--suite",
        str(suite),
        "--tl",
        str(timelimit),
    ]
    completed = subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True
    )
    summaries = json.loads(completed.stdout)
    target_versions = set(versions)
    scores = {
        item["version"]: float(item["relative_score"])
        for item in summaries
        if item["version"] in target_versions
    }
    missing = [version for version in versions if version not in scores]
    if missing:
        raise ValueError(f"stats did not return versions: {', '.join(missing)}")
    return scores


def main() -> int:
    args = parse_args()
    root = repo_root()
    config_path = resolve_path(root, args.config)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")

    name = safe_component(str(config["name"]))
    base_version = str(config["version"])
    base_params_path = resolve_path(root, str(config["base_params"]))
    suite = resolve_path(root, str(config["suite"]))
    timelimit = float(config["timelimit"])
    jobs = int(config.get("jobs", 1))
    steps = config["steps"]

    with base_params_path.open(encoding="utf-8") as f:
        fixed_params = yaml.safe_load(f)
    if not isinstance(fixed_params, dict):
        raise ValueError("base_params must be a YAML mapping")
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")

    output_dir = root / "tuning" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "name": name,
        "version": base_version,
        "base_params": str(base_params_path),
        "suite": str(suite),
        "timelimit": timelimit,
        "jobs": jobs,
        "steps": [],
    }

    for step_index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"step {step_index} must be a mapping")
        step_name = str(step.get("name", f"step-{step_index:02d}"))
        mode, combinations = step_combinations(step, step_index)
        candidates = []
        versions = []
        step_dir = output_dir / f"step-{step_index:02d}"
        print(
            f"step {step_index}/{len(steps)}: {step_name} "
            f"mode={mode} ({len(combinations)} candidates)"
        )

        for candidate_index, values in enumerate(combinations, start=1):
            candidate_version = (
                f"{base_version}-tune-{name}-s{step_index:02d}-c{candidate_index:03d}"
            )
            candidate_params = copy.deepcopy(fixed_params)
            for path, value in values.items():
                set_dotted_value(candidate_params, path, value)
            params_path = step_dir / f"candidate-{candidate_index:03d}.yaml"
            write_yaml(params_path, candidate_params)

            print(
                f"  candidate {candidate_index}/{len(combinations)}: "
                f"{candidate_version} {values}"
            )
            run_candidate(
                root,
                candidate_version,
                params_path,
                suite,
                timelimit,
                jobs,
            )
            versions.append(candidate_version)
            candidates.append(
                {
                    "version": candidate_version,
                    "parameters": values,
                    "params_path": str(params_path),
                    "params": candidate_params,
                }
            )

        scores = load_relative_scores(root, suite, timelimit, versions)
        for candidate in candidates:
            candidate["relative_score"] = scores[candidate["version"]]
        winner = max(candidates, key=lambda candidate: candidate["relative_score"])
        fixed_params = winner["params"]
        write_yaml(output_dir / f"step-{step_index:02d}-best.yaml", fixed_params)

        step_result = {
            "name": step_name,
            "mode": mode,
            "candidates": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "params"
                }
                for candidate in candidates
            ],
            "winner": {
                "version": winner["version"],
                "parameters": winner["parameters"],
                "relative_score": winner["relative_score"],
            },
        }
        result["steps"].append(step_result)
        write_json(output_dir / "results.json", result)
        print(
            f"  winner: {winner['version']} "
            f"relative_score={winner['relative_score']:.6f}"
        )

    best_params_path = output_dir / "best.yaml"
    write_yaml(best_params_path, fixed_params)
    best_version = f"{base_version}-tune-{name}-best"
    compose_solution(root, best_version, best_params_path)
    result["best_version"] = best_version
    write_json(output_dir / "results.json", result)
    print(f"best params: {best_params_path}")
    print(f"best solution: {root / 'solutions' / best_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
