#!/usr/bin/env python3
# ruff: noqa
"""Create an OGC 2026 submission folder under solutions/{version}."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

VERSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Rust solver and compose a submission folder."
    )
    parser.add_argument(
        "version",
        help="Version name used as the output directory: solutions/{version}",
    )
    parser.add_argument(
        "--params",
        required=True,
        help="YAML parameter file to bundle as params.yaml",
    )
    parser.add_argument(
        "--platform",
        choices=("local", "linux"),
        default="local",
        help="Build target platform. default: local",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Build with local Rust logs and panic on collision fallback.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_version(version: str) -> None:
    if not VERSION_RE.fullmatch(version):
        raise ValueError(
            "version must contain only letters, digits, underscore, dot, or hyphen"
        )


def build_solver(root: Path, platform: str, local: bool) -> Path:
    if platform == "local":
        command = ["cargo", "build", "--release"]
        binary = root / "target" / "release" / binary_name()
    elif platform == "linux":
        command = ["cargo", "build", "--release"]
        binary = root / "target" / "release" / binary_name()
    else:
        raise ValueError(f"unsupported platform: {platform}")

    if local:
        command.extend(["--features", "local"])

    print("$ " + " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, cwd=root)

    if completed.returncode != 0:
        raise RuntimeError(f"build failed with code {completed.returncode}")

    if not binary.is_file():
        raise FileNotFoundError(f"built binary not found: {binary}")

    return binary


def binary_name() -> str:
    if sys.platform == "win32":
        return "ogc2026.exe"
    return "ogc2026"


def compose(
    root: Path, version: str, platform: str, params_path: Path, local: bool
) -> Path:
    validate_version(version)
    params_path = params_path.resolve()
    if not params_path.is_file():
        raise FileNotFoundError(f"params file not found: {params_path}")

    source_binary = build_solver(root, platform, local)
    output_dir = root / "solutions" / version

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    solver_path = output_dir / "solver"
    shutil.copy2(source_binary, solver_path)
    solver_path.chmod(solver_path.stat().st_mode | 0o755)

    myalgorithm_path = output_dir / "myalgorithm.py"
    shutil.copy2(root / "tools" / "myalgorithm.py", myalgorithm_path)
    shutil.copy2(params_path, output_dir / "params.yaml")

    return output_dir


def main() -> int:
    args = parse_args()
    root = repo_root()

    try:
        output_dir = compose(
            root, args.version, args.platform, Path(args.params), args.local
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"created: {output_dir}")
    print(f"platform: {args.platform}")
    print("files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
