#!/usr/bin/env python3
"""Run one Cloud Run task shard and upload its logs to Cloud Storage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.cloud import storage  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--timelimit", type=float, required=True)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upload_tree(bucket: storage.Bucket, source: Path, prefix: str) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            bucket.blob(f"{prefix}/{relative}").upload_from_filename(path)


def main() -> int:
    args = parse_args()
    task_index = int(os.environ["CLOUD_RUN_TASK_INDEX"])
    task_count = int(os.environ["CLOUD_RUN_TASK_COUNT"])
    output_prefix = f"runs/{args.run_id}/output/task-{task_index}"
    client = storage.Client()
    bucket = client.bucket(args.bucket)
    status: dict[str, Any] = {
        "task_index": task_index,
        "task_count": task_count,
        "started_at": timestamp(),
        "returncode": 1,
        "cases": [],
        "error": "",
    }

    with tempfile.TemporaryDirectory(prefix="ogc2026-") as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        workspace.mkdir()
        bundle_path = Path(temp_dir) / "bundle.tar.gz"
        returncode = 1

        try:
            bucket.blob(
                f"runs/{args.run_id}/input/bundle.tar.gz"
            ).download_to_filename(bundle_path)
            with tarfile.open(bundle_path, "r:gz") as archive:
                archive.extractall(workspace, filter="data")

            suite_path = workspace / "cloud-suite.json"
            with suite_path.open(encoding="utf-8") as f:
                suite = json.load(f)
            cases = suite["cases"]
            task_cases = cases[task_index::task_count]
            status["cases"] = task_cases

            if task_cases:
                task_suite_path = workspace / f"cloud-suite-task-{task_index}.json"
                task_suite_path.write_text(
                    json.dumps({"cases": task_cases}, ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                command = [
                    sys.executable,
                    "tools/runner.py",
                    args.version,
                    "--suite",
                    task_suite_path.name,
                    "--timelimit",
                    str(args.timelimit),
                ]
                print("$ " + " ".join(command), flush=True)
                returncode = subprocess.run(
                    command, cwd=workspace, check=False
                ).returncode
            else:
                returncode = 0
        except Exception as exc: # noqa
            status["error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            print(status["error"], file=sys.stderr)
        finally:
            status["returncode"] = returncode
            status["finished_at"] = timestamp()
            status_path = Path(temp_dir) / "task-status.json"
            status_path.write_text(
                json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            try:
                upload_tree(bucket, workspace / "log", f"{output_prefix}/log")
                bucket.blob(f"{output_prefix}/task-status.json").upload_from_filename(
                    status_path
                )
            except Exception as exc: # noqa
                print(f"failed to upload task output: {exc}", file=sys.stderr)
                returncode = 1

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
