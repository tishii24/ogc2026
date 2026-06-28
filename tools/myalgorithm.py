# type: ignore

#!/usr/bin/env python3
"""Async Python supervisor for the Rust OGC 2026 solver."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

CANDIDATE_QUEUE_SIZE = 3
RESULT_QUEUE_SIZE = 3
RUST_WORKER_THREADS = 3
PYTHON_NUM_THREADS = 1
RETURN_BUFFER_SECONDS = 0.5
SOLVER_KILL_GRACE_SECONDS = 0.2

THREAD_ENV = {
    "RAYON_NUM_THREADS": str(RUST_WORKER_THREADS),
    "OMP_NUM_THREADS": str(PYTHON_NUM_THREADS),
    "OPENBLAS_NUM_THREADS": str(PYTHON_NUM_THREADS),
    "MKL_NUM_THREADS": str(PYTHON_NUM_THREADS),
    "BLIS_NUM_THREADS": str(PYTHON_NUM_THREADS),
    "VECLIB_MAXIMUM_THREADS": str(PYTHON_NUM_THREADS),
    "NUMEXPR_NUM_THREADS": str(PYTHON_NUM_THREADS),
}


def algorithm(prob_info, timelimit=60):
    check_feasibility = load_check_feasibility()
    return asyncio.run(run_solver(prob_info, float(timelimit), check_feasibility))


def load_check_feasibility():
    here = pathlib.Path(__file__).resolve().parent
    for path in (here, here.parent / "ogc2026" / "alg_tester"):
        if path.is_dir():
            sys.path.insert(0, str(path))
    from utils import check_feasibility

    return check_feasibility


def locate_solver() -> pathlib.Path:
    env_path = os.environ.get("OGC_SOLVER_PATH")
    if env_path:
        return pathlib.Path(env_path).resolve()

    here = pathlib.Path(__file__).resolve().parent
    name = "solver.exe" if sys.platform == "win32" else "solver"
    candidates = [
        here / name,
        here.parent
        / "target"
        / "release"
        / ("ogc2026.exe" if sys.platform == "win32" else "ogc2026"),
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


def solver_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    return env


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def validate_candidate(
    prob_info: dict[str, Any], candidate: dict[str, Any], check_feasibility
):
    result = check_feasibility(prob_info, candidate.get("solution"))
    feasible = bool(result.get("feasible"))
    response: dict[str, Any] = {
        "id": candidate["id"],
        "worker_id": candidate["worker_id"],
        "feasible": feasible,
    }
    objective = result.get("objective")
    if feasible:
        response["objective"] = objective
    return response, candidate.get("solution"), objective


class SolverSupervisor:
    def __init__(self, prob_info: dict[str, Any], timelimit: float, check_feasibility):
        self.prob_info = prob_info
        self.timelimit = timelimit
        self.check_feasibility = check_feasibility
        self.candidate_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=CANDIDATE_QUEUE_SIZE
        )
        self.result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=RESULT_QUEUE_SIZE
        )
        self.best_solution: dict[str, Any] | None = None
        self.best_objective = float("inf")
        self.stop = asyncio.Event()
        self.process: asyncio.subprocess.Process | None = None

    async def run(self) -> dict[str, Any]:
        solver = locate_solver()
        try:
            solver.chmod(solver.stat().st_mode | 0o755)
        except OSError:
            pass

        solver_timelimit = max(0.1, self.timelimit - RETURN_BUFFER_SECONDS)
        self.process = await asyncio.create_subprocess_exec(
            str(solver),
            "--interactive",
            "--timelimit",
            str(solver_timelimit),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(solver.parent),
            env=solver_env(),
        )

        await self.send_initial_problem(solver_timelimit)

        executor = ThreadPoolExecutor(max_workers=1)
        tasks = [
            asyncio.create_task(self.read_candidates()),
            asyncio.create_task(self.validate_loop(executor)),
            asyncio.create_task(self.write_results()),
            asyncio.create_task(self.drain_stderr()),
        ]

        try:
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=solver_timelimit)
            except asyncio.TimeoutError:
                pass
        finally:
            self.stop.set()
            await self.stop_solver()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            executor.shutdown(wait=False, cancel_futures=True)

        if self.best_solution is None:
            raise RuntimeError("no feasible solution was validated")
        return self.best_solution

    async def send_initial_problem(self, solver_timelimit: float) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("solver stdin is not available")
        message = {"problem": self.prob_info, "timelimit": solver_timelimit}
        self.process.stdin.write((compact_json(message) + "\n").encode())
        await self.process.stdin.drain()

    async def read_candidates(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while not self.stop.is_set():
            line = await self.process.stdout.readline()
            if not line:
                self.stop.set()
                return
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"invalid solver candidate json: {exc}", file=sys.stderr)
                continue
            await self.candidate_queue.put(candidate)

    async def validate_loop(self, executor: ThreadPoolExecutor) -> None:
        loop = asyncio.get_running_loop()
        while not self.stop.is_set():
            candidate = await self.candidate_queue.get()
            try:
                response, solution, objective = await loop.run_in_executor(
                    executor,
                    validate_candidate,
                    self.prob_info,
                    candidate,
                    self.check_feasibility,
                )
            except Exception as exc:
                response = {
                    "id": candidate.get("id"),
                    "worker_id": candidate.get("worker_id"),
                    "feasible": False,
                }
                print(f"validation failed: {exc}", file=sys.stderr)
                solution = None
                objective = None

            if response["feasible"] and objective is not None:
                if float(objective) < self.best_objective:
                    self.best_objective = float(objective)
                    self.best_solution = solution
            await self.result_queue.put(response)

    async def write_results(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        while not self.stop.is_set():
            result = await self.result_queue.get()
            try:
                self.process.stdin.write((compact_json(result) + "\n").encode())
                await self.process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                self.stop.set()
                return

    async def drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while not self.stop.is_set():
            line = await self.process.stderr.readline()
            if not line:
                return
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()

    async def stop_solver(self) -> None:
        if self.process is None:
            return
        if self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(
                self.process.wait(), timeout=SOLVER_KILL_GRACE_SECONDS
            )
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()


async def run_solver(
    prob_info: dict[str, Any], timelimit: float, check_feasibility
) -> dict[str, Any]:
    supervisor = SolverSupervisor(prob_info, timelimit, check_feasibility)
    return await supervisor.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tools/myalgorithm.py standalone.")
    parser.add_argument("input", help="Problem JSON path")
    parser.add_argument("timelimit", nargs="?", type=float, default=60.0)
    parser.add_argument(
        "--output", help="Write solution JSON to this path instead of stdout"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    with open(args.input, encoding="utf-8") as f:
        prob_info = json.load(f)
    solution = algorithm(prob_info, args.timelimit)
    output = compact_json(solution) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    print(f"elapsed={time.time() - start:.3f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
