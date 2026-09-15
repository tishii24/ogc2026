#!/usr/bin/env python3
"""Build a Linux solution and run a suite with Cloud Run Jobs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore

VERSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Create or update Google Cloud resources.")
    setup.add_argument("--project", help="Google Cloud project ID.")
    setup.add_argument("--bucket", help="Cloud Storage bucket name.")
    setup.add_argument("--region", help="Google Cloud region.")

    subparsers.add_parser(
        "update", help="Update an existing Cloud Run Job from tools/cloud/config.yaml."
    )

    run = subparsers.add_parser("run", help="Run a suite on Cloud Run Jobs.")
    run.add_argument("version", help="Version directory under solutions/.")
    run.add_argument("--params", default="params/default.yaml")
    run.add_argument("--suite", required=True)
    run.add_argument("--timelimit", type=float, required=True)
    run.add_argument(
        "--local",
        action="store_true",
        help="Build the cloud solver with the Rust local feature.",
    )

    logs = subparsers.add_parser("logs", help="Tail logs for a Cloud Run Job execution.")
    logs.add_argument(
        "execution",
        nargs="?",
        help="Execution name. Defaults to the latest execution.",
    )

    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("$ " + shlex.join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def command_succeeds(command: list[str], *, cwd: Path) -> bool:
    return (
        subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def load_config(root: Path) -> dict[str, Any]:
    config_path = root / "tools" / "cloud" / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("tools/cloud/config.yaml must be a mapping")# noqa

    local_path = root / "tools" / "cloud" / "config.local.yaml"
    if local_path.is_file():
        with local_path.open(encoding="utf-8") as f:
            local = yaml.safe_load(f)
        if not isinstance(local, dict):
            raise ValueError("tools/cloud/config.local.yaml must be a mapping")
        config.update(local)
    return config


def active_project(root: Path) -> str:
    completed = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    project = completed.stdout.strip()
    if not project or project == "(unset)":
        raise ValueError("Google Cloud project is not set")
    return project


def image_url(config: dict[str, Any]) -> str:
    return (
        f"{config['region']}-docker.pkg.dev/{config['project']}/"
        f"{config['repository']}/{config['base_image']}:latest"
    )


def save_local_config(root: Path, project: str, bucket: str, region: str) -> None:
    path = root / "tools" / "cloud" / "config.local.yaml"
    path.write_text(
        yaml.safe_dump(
            {"project": project, "bucket": bucket, "region": region},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def setup_cloud(root: Path, args: argparse.Namespace) -> int:
    config = load_config(root)
    configured_project = config.get("project")
    project = args.project or configured_project or active_project(root)
    region = args.region or config["region"]
    configured_bucket = (
        config.get("bucket")
        if args.project is None or args.project == configured_project
        else None
    )
    bucket = args.bucket or configured_bucket or f"{project}-ogc2026-runs"
    config.update({"project": project, "bucket": bucket, "region": region})
    save_local_config(root, project, bucket, region)

    run_command(
        [
            "gcloud",
            "services",
            "enable",
            "artifactregistry.googleapis.com",
            "run.googleapis.com",
            "cloudbuild.googleapis.com",
            "storage.googleapis.com",
            "--project",
            project,
        ],
        cwd=root,
    )

    repository = str(config["repository"])
    describe_repository = [
        "gcloud",
        "artifacts",
        "repositories",
        "describe",
        repository,
        "--location",
        region,
        "--project",
        project,
    ]
    if not command_succeeds(describe_repository, cwd=root):
        run_command(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "create",
                repository,
                "--repository-format",
                "docker",
                "--location",
                region,
                "--project",
                project,
            ],
            cwd=root,
        )

    bucket_uri = f"gs://{bucket}"
    if not command_succeeds(
        ["gcloud", "storage", "buckets", "describe", bucket_uri, "--project", project],
        cwd=root,
    ):
        run_command(
            [
                "gcloud",
                "storage",
                "buckets",
                "create",
                bucket_uri,
                "--location",
                region,
                "--uniform-bucket-level-access",
                "--project",
                project,
            ],
            cwd=root,
        )

    service_account = str(config["service_account"])
    service_account_email = f"{service_account}@{project}.iam.gserviceaccount.com"
    if not command_succeeds(
        [
            "gcloud",
            "iam",
            "service-accounts",
            "describe",
            service_account_email,
            "--project",
            project,
        ],
        cwd=root,
    ):
        run_command(
            [
                "gcloud",
                "iam",
                "service-accounts",
                "create",
                service_account,
                "--display-name",
                "OGC 2026 Cloud Runner",
                "--project",
                project,
            ],
            cwd=root,
        )

    run_command(
        [
            "gcloud",
            "storage",
            "buckets",
            "add-iam-policy-binding",
            bucket_uri,
            "--member",
            f"serviceAccount:{service_account_email}",
            "--role",
            "roles/storage.objectUser",
            "--project",
            project,
        ],
        cwd=root,
    )

    image = image_url(config)
    run_command(
        [
            "gcloud",
            "builds",
            "submit",
            "tools/cloud",
            "--tag",
            image,
            "--project",
            project,
            "--region",
            region,
        ],
        cwd=root,
    )

    run_command(
        [
            "gcloud",
            "run",
            "jobs",
            "deploy",
            str(config["job"]),
            "--image",
            image,
            "--region",
            region,
            "--project",
            project,
            "--service-account",
            service_account_email,
            "--tasks",
            str(config["tasks"]),
            "--parallelism",
            str(config["tasks"]),
            "--cpu",
            str(config["cpu"]),
            "--memory",
            str(config["memory"]),
            "--task-timeout",
            str(config["task_timeout"]),
            "--max-retries",
            "0",
            "--command",
            "python3.12",
            "--args",
            "/opt/ogc/task_runner.py",
            "--quiet",
        ],
        cwd=root,
    )

    run_command(
        [
            "gcloud",
            "auth",
            "configure-docker",
            f"{region}-docker.pkg.dev",
            "--quiet",
        ],
        cwd=root,
    )
    print(f"config: {root / 'tools' / 'cloud' / 'config.local.yaml'}")
    print(f"image: {image}")
    return 0


def update_cloud(root: Path) -> int:
    config = load_config(root)
    project = config.get("project")
    if not project:
        raise ValueError("cloud setup is incomplete; run cloud_runner.py setup")

    service_account = f"{config['service_account']}@{project}.iam.gserviceaccount.com"
    run_command(
        [
            "gcloud",
            "run",
            "jobs",
            "update",
            str(config["job"]),
            "--region",
            str(config["region"]),
            "--project",
            str(project),
            "--service-account",
            service_account,
            "--tasks",
            str(config["tasks"]),
            "--parallelism",
            str(config["tasks"]),
            "--cpu",
            str(config["cpu"]),
            "--memory",
            str(config["memory"]),
            "--task-timeout",
            str(config["task_timeout"]),
            "--max-retries",
            "0",
        ],
        cwd=root,
    )
    return 0


def path_in_root(root: Path, value: str) -> tuple[Path, Path]:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must be under the repository: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, relative


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    ]


def collect_suite_cases(root: Path, suite_path: Path) -> list[Path]:
    with suite_path.open(encoding="utf-8") as f:
        suite = json.load(f)
    if not isinstance(suite, dict):
        raise ValueError("suite must be a JSON object") # noqa
    patterns = suite.get("cases")
    if not isinstance(patterns, list) or not all(
        isinstance(pattern, str) for pattern in patterns
    ):
        raise ValueError("suite.cases must be a list of strings")

    cases: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        source = Path(pattern)
        glob_pattern = str(source if source.is_absolute() else root / source)
        matches = sorted(
            (Path(match).resolve() for match in glob.glob(glob_pattern)),
            key=natural_key,
        )
        if not matches:
            candidate = source if source.is_absolute() else root / source
            matches = [candidate.resolve()]
        for case in matches:
            try:
                case.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"case must be under the repository: {case}") from exc
            if not case.is_file():
                raise FileNotFoundError(case)
            if case not in seen:
                cases.append(case)
                seen.add(case)
    if not cases:
        raise ValueError("suite contains no cases")
    return cases


def compose_linux_solution(
    root: Path,
    config: dict[str, Any],
    version: str,
    params_relative: Path,
    local: bool,
) -> Path:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{root}:/work/ogc2026",
        "-w",
        "/work/ogc2026",
        image_url(config),
        "python3.12",
        "tools/composer.py",
        version,
        "--params",
        params_relative.as_posix(),
        "--platform",
        "linux",
    ]
    if local:
        command.append("--local")
    run_command(command, cwd=root)
    return root / "solutions" / version


def create_bundle(
    root: Path,
    bundle_path: Path,
    solution_dir: Path,
    cases: list[Path],
) -> None:
    manifest_path = bundle_path.parent / "cloud-suite.json"
    manifest_path.write_text(
        json.dumps(
            {"cases": [case.relative_to(root).as_posix() for case in cases]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with tarfile.open(bundle_path, "w:gz") as archive:
        archive.add(solution_dir, arcname=solution_dir.relative_to(root))
        archive.add(root / "tools" / "runner.py", arcname="tools/runner.py")
        checker_dir = root / "tools" / "alg_tester"
        archive.add(checker_dir, arcname=checker_dir.relative_to(root))
        for case in cases:
            archive.add(case, arcname=case.relative_to(root))
        archive.add(manifest_path, arcname="cloud-suite.json")


def append_score_shards(root: Path, shard_paths: list[Path]) -> None:
    if not shard_paths:
        return
    score_path = root / "log" / "score.csv"
    score_path.parent.mkdir(exist_ok=True)
    write_header = not score_path.exists() or score_path.stat().st_size == 0

    with score_path.open("a", newline="", encoding="utf-8") as output:
        writer: csv.DictWriter[str] | None = None
        for shard_path in sorted(shard_paths):
            with shard_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None:
                    continue
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                elif reader.fieldnames != writer.fieldnames:
                    raise ValueError(f"score columns differ: {shard_path}")
                writer.writerows(reader)


def merge_downloaded_logs(root: Path, download_dir: Path) -> list[dict[str, Any]]:
    shard_scores = list(download_dir.rglob("score.csv"))
    append_score_shards(root, shard_scores)

    local_log = root / "log"
    local_log.mkdir(exist_ok=True)
    for shard_score in shard_scores:
        shard_log = shard_score.parent
        for child in shard_log.iterdir():
            if child.name == "score.csv":
                continue
            destination = local_log / child.name
            if child.is_dir():
                shutil.copytree(child, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(child, destination)

    statuses = []
    for status_path in sorted(download_dir.rglob("task-status.json")):
        with status_path.open(encoding="utf-8") as f:
            statuses.append(json.load(f))
    return statuses


def tail_logs(root: Path, args: argparse.Namespace) -> int:
    config = load_config(root)
    if not config.get("project"):
        raise ValueError("cloud setup is incomplete; run cloud_runner.py setup")

    execution = args.execution
    if execution is None:
        completed = run_command(
            [
                "gcloud",
                "run",
                "jobs",
                "executions",
                "list",
                "--job",
                str(config["job"]),
                "--region",
                str(config["region"]),
                "--project",
                str(config["project"]),
                "--sort-by=~metadata.creationTimestamp",
                "--limit=1",
                "--format=value(metadata.name)",
            ],
            cwd=root,
            capture_output=True,
        )
        execution = completed.stdout.strip()
        if not execution:
            raise ValueError(f"no executions found for job {config['job']}")

    print(f"execution: {execution}", flush=True)
    completed = run_command(
        [
            "gcloud",
            "beta",
            "run",
            "jobs",
            "executions",
            "logs",
            "tail",
            execution,
            "--region",
            str(config["region"]),
            "--project",
            str(config["project"]),
        ],
        cwd=root,
        check=False,
    )
    return completed.returncode


def run_cloud(root: Path, args: argparse.Namespace) -> int:
    if not VERSION_RE.fullmatch(args.version):
        raise ValueError(
            "version must contain only letters, digits, underscore, dot, or hyphen"
        )
    if args.timelimit <= 0:
        raise ValueError("timelimit must be positive")

    config = load_config(root)
    if not config.get("project") or not config.get("bucket"):
        raise ValueError("cloud setup is incomplete; run cloud_runner.py setup")

    update_cloud(root)

    _, params_relative = path_in_root(root, args.params)
    suite_path, _ = path_in_root(root, args.suite)
    cases = collect_suite_cases(root, suite_path)
    solution_dir = compose_linux_solution(
        root, config, args.version, params_relative, args.local
    )

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        + f"-{args.version}-{uuid.uuid4().hex[:8]}"
    )
    bucket = str(config["bucket"])
    remote_root = f"gs://{bucket}/runs/{run_id}"

    with tempfile.TemporaryDirectory(prefix="ogc2026-cloud-") as temp_dir:
        temp = Path(temp_dir)
        bundle_path = temp / "bundle.tar.gz"
        create_bundle(root, bundle_path, solution_dir, cases)
        run_command(
            [
                "gcloud",
                "storage",
                "cp",
                str(bundle_path),
                f"{remote_root}/input/bundle.tar.gz",
                "--project",
                str(config["project"]),
            ],
            cwd=root,
        )

        container_args = [
            "/opt/ogc/task_runner.py",
            "--bucket",
            bucket,
            "--run-id",
            run_id,
            "--version",
            args.version,
            "--timelimit",
            str(args.timelimit),
        ]
        execution = run_command(
            [
                "gcloud",
                "run",
                "jobs",
                "execute",
                str(config["job"]),
                "--region",
                str(config["region"]),
                "--project",
                str(config["project"]),
                "--tasks",
                str(config["tasks"]),
                "--wait",
                f"--args={','.join(container_args)}",
            ],
            cwd=root,
            check=False,
        )

        download_dir = temp / "output"
        download_dir.mkdir()
        downloaded = run_command(
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                f"{remote_root}/output",
                str(download_dir),
                "--project",
                str(config["project"]),
            ],
            cwd=root,
            check=False,
        )
        if downloaded.returncode != 0:
            print(f"failed to download logs; remote files: {remote_root}", file=sys.stderr)
            return 1

        statuses = merge_downloaded_logs(root, download_dir)
        for status in sorted(statuses, key=lambda item: item["task_index"]):
            print(
                f"task {status['task_index']}: returncode={status['returncode']} "
                f"cases={len(status['cases'])}"
            )
            if status.get("error"):
                print(f"  error: {status['error']}")

        run_command(
            [
                "gcloud",
                "storage",
                "rm",
                "--recursive",
                remote_root,
                "--project",
                str(config["project"]),
            ],
            cwd=root,
        )

    failed = (
        execution.returncode != 0
        or len(statuses) != int(config["tasks"])
        or any(status["returncode"] != 0 for status in statuses)
    )
    print(f"log: {root / 'log'}")
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    root = repo_root()
    try:
        if args.command == "setup":
            return setup_cloud(root, args)
        if args.command == "update":
            return update_cloud(root)
        if args.command == "logs":
            return tail_logs(root, args)
        return run_cloud(root, args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
