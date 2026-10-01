# SPDX-License-Identifier: Apache-2.0
"""Run synchronized realtime sessions and summarize client-side serving behavior."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.client import SAMPLE_RATE, run_session
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName
from benchmarks.duplex.serving_metrics import (
    LATE_SEND_THRESHOLD_S,
    distribution,
    empty_session_metrics,
    session_metrics,
)
from benchmarks.duplex.serving_summary import aggregate_sessions
from benchmarks.duplex.v15_audio import normalize_audio
from benchmarks.runtime_metrics import ResourceMonitor, collect_benchmark_provenance

START_LEAD_S = 0.2
DEFAULT_RESERVE_MS = 80.0
LOOP_TICK_S = 0.01
DEFAULT_LOOP_LAG_LIMIT_MS = 20.0


async def run_concurrency(
    url: str,
    pcm: bytes,
    *,
    concurrency: int,
    profile: ProfileName,
    output_dir: Path,
    timeout_s: float,
    reserve_s: float,
    stagger_s: float = 0.0,
    loop_lag_limit_s: float = DEFAULT_LOOP_LAG_LIMIT_MS / 1000,
) -> dict[str, JsonValue]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not pcm or len(pcm) % 2:
        raise ValueError("input must be nonempty PCM16")
    if stagger_s < 0 or loop_lag_limit_s <= 0:
        raise ValueError("stagger must be nonnegative and loop lag limit positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    loop_lags_s: list[float] = []

    async def measure_loop_lag() -> None:
        while True:
            deadline_s = time.perf_counter() + LOOP_TICK_S
            await asyncio.sleep(LOOP_TICK_S)
            loop_lags_s.append(max(0.0, time.perf_counter() - deadline_s))

    ticker = asyncio.create_task(measure_loop_lag())
    loop = asyncio.get_running_loop()
    start_gate: asyncio.Future[float] = loop.create_future()
    ready = [loop.create_future() for _ in range(concurrency)]
    trace_paths = []
    tasks = []
    for index in range(concurrency):
        session_dir = output_dir / f"session-{index:03d}"
        session_dir.mkdir()
        trace_path = session_dir / "trace.jsonl"
        trace_paths.append(trace_path)
        task = asyncio.create_task(
            run_session(
                url,
                pcm,
                scenario="continuous",
                trace_path=trace_path,
                timeout_s=timeout_s,
                profile=profile,
                start_gate=start_gate,
                ready=ready[index],
                start_offset_s=stagger_s * index / concurrency,
            )
        )
        task.add_done_callback(
            lambda completed, readiness=ready[index]: (
                readiness.set_result(False) if not readiness.done() else None
            )
        )
        tasks.append(task)
    configured = await asyncio.gather(*ready)
    start_s = time.perf_counter() + START_LEAD_S
    start_gate.set_result(start_s)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ticker.cancel()
    await asyncio.gather(ticker, return_exceptions=True)
    duration_s = len(pcm) / (2 * SAMPLE_RATE)
    sessions = []
    for index, (trace_path, result) in enumerate(zip(trace_paths, results)):
        session_id = f"session-{index:03d}"
        try:
            session = session_metrics(
                trace_path,
                session_id=session_id,
                input_duration_s=duration_s,
                profile=profile,
                reserve_s=reserve_s,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            session = empty_session_metrics(trace_path, session_id, duration_s, profile)
            session["errors"].append(f"unreadable session artifacts: {exc}")
        if isinstance(result, BaseException):
            session["success"] = False
            session["errors"].append(f"client task: {type(result).__name__}: {result}")
        sessions.append(session)
    summary: dict[str, JsonValue] = {
        "concurrency": concurrency,
        "input_duration_s": duration_s,
        "common_start_s": start_s,
        "stagger_s": stagger_s,
        "configured_sessions": sum(configured),
        "profile": profile,
        "playback_startup_reserve_s": reserve_s,
        "sessions": sessions,
        "aggregate": aggregate_sessions(sessions),
        "loop_lag_s": distribution(loop_lags_s),
        "loop_lag_limit_s": loop_lag_limit_s,
        "send_lateness_limit_s": LATE_SEND_THRESHOLD_S,
        "client_timing_valid": bool(loop_lags_s)
        and distribution(loop_lags_s)["p99"] <= loop_lag_limit_s
        and all(
            session["send_lateness_s"]["p99"] is not None
            and session["send_lateness_s"]["p99"] <= LATE_SEND_THRESHOLD_S
            for session in sessions
            if session["admitted"]
        )
        and any(session["admitted"] for session in sessions),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def format_ms(value: float | None) -> str:
    return f"{value * 1000:.1f}" if value is not None else "-"


def format_percent(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "-"


def print_summary(summary: dict[str, JsonValue]) -> None:
    aggregate = summary["aggregate"]
    print(
        f"Model: {summary['profile']}  Concurrency: {summary['concurrency']}  "
        f"Duration: {summary['input_duration_s']:.2f}s  "
        f"Admitted: {aggregate['admitted_sessions']}/{aggregate['attempted_sessions']}  "
        f"Rejected: {aggregate['rejected_sessions']}  "
        f"Success: {aggregate['successful_admitted_sessions']}/{aggregate['admitted_sessions']} admitted"
    )
    print(
        f"Loop lag p99: {format_ms(summary['loop_lag_s']['p99'])}ms "
        f"(n={summary['loop_lag_s']['n']}, "
        f"valid={summary['client_timing_valid']})"
    )
    print(f"{'Metric':22} {'p50':>9} {'p75':>9} {'p95':>9} {'p99':>9} {'max':>9}")
    for label, name in (
        ("Unit lag (ms)", "unit_lag_s"),
        ("Response TTFA (ms)", "response_ttfa_s"),
        ("Response gap excess", "response_gap_excess_s"),
        ("Response max drift", "response_max_drift_s"),
        ("Response buffer (ms)", "response_required_playout_buffer_s"),
        ("Session TTFA (ms)", "ttfa_s"),
        ("Send lateness (ms)", "send_lateness_s"),
        ("Output gap (ms)", "output_gap_s"),
        ("Gap excess (ms)", "output_gap_excess_s"),
        ("Output drift (ms)", "output_drift_s"),
        ("Media schedule lag (ms)", "media_schedule_lag_s"),
        ("Media send lag (ms)", "media_send_lag_s"),
    ):
        values = aggregate[name]
        print(
            f"{label:22} {format_ms(values['p50']):>9} "
            f"{format_ms(values['p75']):>9} "
            f"{format_ms(values['p95']):>9} "
            f"{format_ms(values['p99']):>9} {format_ms(values['max']):>9} "
            f"n={values['n']}"
        )
    coverage = aggregate["output_coverage"]["p50"]
    print(f"Output coverage p50: {format_percent(coverage)}")
    print(
        f"Late sends (>{aggregate['late_send_threshold_s'] * 1000:.0f}ms): "
        f"{aggregate['late_send_count']}/{aggregate['send_lateness_s']['n']} "
        f"({format_percent(aggregate['late_send_rate'])})"
    )
    print(
        "Final output drift p50/p95: "
        f"{format_ms(aggregate['final_output_drift_s']['p50'])}/"
        f"{format_ms(aggregate['final_output_drift_s']['p95'])}ms"
    )
    print(
        "Required playout buffer p50/p95: "
        f"{format_ms(aggregate['required_playout_buffer_s']['p50'])}/"
        f"{format_ms(aggregate['required_playout_buffer_s']['p95'])}ms "
        f"(n={aggregate['required_playout_buffer_s']['n']})"
    )
    print(
        f"Underrun sessions: {aggregate['underrun_sessions'] if aggregate['underrun_sessions'] is not None else '-'}/"
        f"{aggregate['playout_sessions']}  "
        f"Count: {aggregate['underrun_count'] if aggregate['underrun_count'] is not None else '-'}  "
        f"Total/max: {format_ms(aggregate['underrun_total_s'])}/"
        f"{format_ms(aggregate['underrun_worst_s'])}ms  "
        f"Ratio: {format_percent(aggregate['underrun_ratio'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--profile", choices=list(PROFILES), default=DEFAULT_PROFILE)
    concurrency = parser.add_mutually_exclusive_group(required=True)
    concurrency.add_argument("--concurrency", type=int)
    concurrency.add_argument("--concurrencies")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--startup-reserve-ms", type=float, default=DEFAULT_RESERVE_MS)
    parser.add_argument("--stagger-ms", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--loop-lag-limit-ms", type=float, default=DEFAULT_LOOP_LAG_LIMIT_MS
    )
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--gpu-process-pid", type=int, action="append")
    parser.add_argument("--model-id", default="unspecified")
    parser.add_argument("--model-revision")
    parser.add_argument("--server-sha")
    parser.add_argument("--server-config", type=Path)
    args = parser.parse_args()
    profile: ProfileName = args.profile
    levels = (
        [args.concurrency]
        if args.concurrency is not None
        else [int(value) for value in args.concurrencies.split(",")]
    )
    if not levels or any(level < 1 for level in levels):
        parser.error("concurrency values must be positive")
    if len(set(levels)) != len(levels):
        parser.error("concurrency values must be unique")
    if args.timeout_s <= 0 or args.startup_reserve_ms < 0:
        parser.error("timeout must be positive and startup reserve nonnegative")
    if args.stagger_ms < 0 or args.repeats < 1 or args.loop_lag_limit_ms <= 0:
        parser.error("stagger must be nonnegative; repeats and loop lag limit positive")
    if args.gpu_process_pid and args.gpu_index is None:
        parser.error("--gpu-process-pid requires --gpu-index")
    if args.gpu_index is not None and args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    if args.gpu_process_pid and any(pid <= 0 for pid in args.gpu_process_pid):
        parser.error("--gpu-process-pid must be positive")
    server_config = {}
    if args.server_config is not None:
        server_config = json.loads(args.server_config.read_text(encoding="utf-8"))
        if not isinstance(server_config, dict):
            parser.error("server config must be a JSON object")
    if args.server_sha is not None:
        server_config["server_sha"] = args.server_sha
    pcm, _ = normalize_audio(args.audio)
    output_dir = args.output_dir or Path("serving-results") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)

    async def run_all() -> (
        tuple[list[dict[str, JsonValue]], dict[str, JsonValue] | None]
    ):
        await run_concurrency(
            args.url,
            pcm,
            concurrency=1,
            profile=profile,
            output_dir=output_dir / "warmup",
            timeout_s=args.timeout_s,
            reserve_s=args.startup_reserve_ms / 1000,
        )
        monitor = (
            ResourceMonitor(
                gpu_index=args.gpu_index,
                gpu_process_pids=args.gpu_process_pid,
            ).start()
            if args.gpu_index is not None
            else None
        )
        summaries = []
        try:
            for repeat in range(args.repeats):
                for level in levels:
                    summaries.append(
                        await run_concurrency(
                            args.url,
                            pcm,
                            concurrency=level,
                            profile=profile,
                            output_dir=output_dir
                            / f"repeat-{repeat + 1}"
                            / f"c{level}",
                            timeout_s=args.timeout_s,
                            reserve_s=args.startup_reserve_ms / 1000,
                            stagger_s=args.stagger_ms / 1000,
                            loop_lag_limit_s=args.loop_lag_limit_ms / 1000,
                        )
                    )
                    summaries[-1]["repeat"] = repeat + 1
        finally:
            resources = monitor.stop() if monitor is not None else None
        return summaries, resources

    summaries, resources = asyncio.run(run_all())
    provenance = collect_benchmark_provenance(
        model_id=args.model_id,
        model_revision=args.model_revision,
        dataset_id=str(args.audio),
        dataset_revision=None,
        launch_command=None,
        server_config=server_config,
        evaluation_input_sha256=hashlib.sha256(pcm).hexdigest(),
    )
    level_summaries = [
        {
            "concurrency": level,
            "repeats": args.repeats,
            "client_timing_valid": all(
                run["client_timing_valid"]
                for run in summaries
                if run["concurrency"] == level
            ),
            "loop_lag_p99_s": max(
                run["loop_lag_s"]["p99"]
                for run in summaries
                if run["concurrency"] == level
            ),
            "aggregate": aggregate_sessions(
                [
                    session
                    for run in summaries
                    if run["concurrency"] == level
                    for session in run["sessions"]
                ]
            ),
        }
        for level in levels
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": summaries,
                "levels": level_summaries,
                "provenance": provenance,
                "resources": resources,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if len(summaries) == 1:
        print_summary(summaries[0])
    else:
        print(
            "Model  C  repeats  admitted/attempted  success/admitted  unit lag p95(n)  "
            "response TTFA p95(n)  response gap-excess p99(n)  session TTFA p95(n)  coverage p50(n)  "
            "send lateness p95/p99(ms)  loop lag p99(ms)  "
            "late sends  buffer p95(n)  client timing valid  underrun sessions  underrun ratio"
        )
        for summary in level_summaries:
            aggregate = summary["aggregate"]
            underrun_sessions = f"{aggregate['underrun_sessions'] if aggregate['underrun_sessions'] is not None else '-'}/{aggregate['playout_sessions']}"
            print(
                f"{profile} {summary['concurrency']:<2} {summary['repeats']:<7} "
                f"{aggregate['admitted_sessions']}/{aggregate['attempted_sessions']} "
                f"{aggregate['successful_admitted_sessions']}/{aggregate['admitted_sessions']} "
                f"{format_ms(aggregate['unit_lag_s']['p95'])}({aggregate['unit_lag_s']['n']}) "
                f"{format_ms(aggregate['response_ttfa_s']['p95'])}({aggregate['response_ttfa_s']['n']}) "
                f"{format_ms(aggregate['response_gap_excess_s']['p99'])}({aggregate['response_gap_excess_s']['n']}) "
                f"{format_ms(aggregate['ttfa_s']['p95'])}({aggregate['ttfa_s']['n']}) "
                f"{format_percent(aggregate['output_coverage']['p50'])}({aggregate['output_coverage']['n']}) "
                f"{format_ms(aggregate['send_lateness_s']['p95'])}/{format_ms(aggregate['send_lateness_s']['p99'])} "
                f"{format_ms(summary['loop_lag_p99_s'])} "
                f"{format_percent(aggregate['late_send_rate']):>11} "
                f"{format_ms(aggregate['required_playout_buffer_s']['p95'])}"
                f"({aggregate['required_playout_buffer_s']['n']}) "
                f"{str(summary['client_timing_valid']):>10} "
                f"{underrun_sessions:>17} "
                f"{format_percent(aggregate['underrun_ratio']):>15}"
            )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
