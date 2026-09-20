"""Experiment-2 open-loop arrivals around the unchanged SeedTTS client.

Run with the usual benchmark CLI arguments plus --concurrency 0 and a finite
--request-rate. Saves arrivals.json beside speed_results.json. Warmup, payloads,
request UUIDs and generation parameters remain owned by the upstream client.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np


def poisson_offsets_ns(count: int, rate: float, seed: int = 42) -> list[int]:
    if count < 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("A finite positive request rate and nonnegative count are required")
    # Match the legacy seeded NumPy distribution without sharing mutable RNG
    # state with warmup, model sampling, or another comparison arm.
    intervals = np.random.RandomState(seed).exponential(1.0 / rate, size=count)
    return np.rint(np.cumsum(intervals) * 1e9).astype(np.int64).tolist()


async def dispatch_arrivals(
    runner: Any,
    session: Any,
    samples: list[Any],
    send_fn: Any,
    output_path: Path,
    *,
    seed: int = 42,
    capture: Callable[[dict], None] | None = None,
) -> list[Any]:
    if runner.config.max_concurrency != 0:
        raise ValueError("Experiment-2 arrival capture requires --concurrency 0")
    offsets = poisson_offsets_ns(len(samples), runner.config.request_rate, seed)
    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Arrival capture requires unique sample IDs")
    if output_path.exists():
        raise FileExistsError(output_path)
    epoch_ns = time.time_ns()
    origin_ns = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    tasks: list[asyncio.Task] = []
    status = "incomplete"

    async def send(sample: Any, row: dict[str, Any]) -> Any:
        row["task_start_ns"] = time.time_ns()
        row["task_start_monotonic_ns"] = time.perf_counter_ns()
        try:
            result = await send_fn(session, sample)
            row.update(
                request_id=result.request_id,
                server_request_id=result.server_request_id,
                worker_id=result.worker_id,
                send_ns=result.request_start_ns,
                end_ns=result.request_end_ns,
                is_success=result.is_success,
                error=result.error,
            )
            return result
        except BaseException as exc:
            row["dispatch_error"] = repr(exc)
            raise
        finally:
            row["task_end_ns"] = time.time_ns()
            row["task_end_monotonic_ns"] = time.perf_counter_ns()

    try:
        for index, (sample, offset) in enumerate(zip(samples, offsets)):
            deadline = origin_ns + offset
            await asyncio.sleep(max(0, deadline - time.perf_counter_ns()) / 1e9)
            dispatched = time.perf_counter_ns()
            row = {
                "index": index,
                "sample_id": sample.sample_id,
                "nominal_offset_ns": offset,
                "nominal_arrival_ns": epoch_ns + offset,
                "nominal_arrival_monotonic_ns": deadline,
                "dispatch_ns": time.time_ns(),
                "dispatch_monotonic_ns": dispatched,
                "dispatch_lag_ns": dispatched - deadline,
            }
            rows.append(row)
            tasks.append(asyncio.create_task(send(sample, row)))
        results = list(await asyncio.gather(*tasks))
        status = "collected"
        return results
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        completed_ns = time.perf_counter_ns()
        schedule = {"sample_ids": sample_ids, "offsets_ns": offsets}
        payload = {
            "status": status,
            "seed": seed,
            "requested_rate": runner.config.request_rate,
            "expected_samples": len(samples),
            "concurrency": 0,
            "warmup": "Upstream warmup precedes this recorded dispatch window",
            "origin_wall_ns": epoch_ns,
            "origin_monotonic_ns": origin_ns,
            "dispatch_window_end_monotonic_ns": completed_ns,
            "schedule_sha256": hashlib.sha256(
                json.dumps(schedule, separators=(",", ":")).encode()
            ).hexdigest(),
            "schedule": schedule,
            "per_request": rows,
        }
        if capture is None:
            save_capture(output_path, payload)
        else:
            capture(payload)


def save_capture(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True)
    known, _ = parser.parse_known_args()
    output_path = Path(known.output_dir) / "arrivals.json"
    from benchmarks.benchmarker.runner import BenchmarkRunner
    from benchmarks.eval.benchmark_tts_seedtts import main as benchmark_main

    captures = []

    async def dispatch(self: Any, session: Any, samples: list, send_fn: Any) -> list:
        return await dispatch_arrivals(
            self, session, samples, send_fn, output_path, capture=captures.append
        )

    BenchmarkRunner._dispatch = dispatch
    np.random.seed(42)
    try:
        benchmark_main()
    finally:
        # NFS/JSON persistence must not extend BenchmarkRunner.wall_clock_s.
        for payload in captures:
            save_capture(output_path, payload)


if __name__ == "__main__":
    main()
