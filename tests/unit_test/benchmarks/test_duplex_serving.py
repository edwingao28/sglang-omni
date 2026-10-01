# SPDX-License-Identifier: Apache-2.0
"""CPU-only serving timing and coordinated session tests."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path
from unittest.mock import patch

import pytest
import websockets
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request, Response

from benchmarks.duplex.client import (
    PACKET_BYTES,
    SEND_RECEIPTS_FILE,
    run_session,
    scheduled_send_s,
)
from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName
from benchmarks.duplex.serving import print_summary, run_concurrency
from benchmarks.duplex.serving_metrics import (
    distribution,
    empty_session_metrics,
    session_metrics,
)
from benchmarks.duplex.serving_summary import aggregate_sessions
from tests.unit_test.benchmarks.test_duplex_client import DuplexPeer

INPUT_DURATION_S = 0.32


def recorded_session(
    trace_path: Path,
    output_times_s: list[float],
    *,
    send_delay_s: float = 0.0,
    failed: bool = False,
    output_samples: list[int] | None = None,
    media_time: bool = False,
) -> dict[str, JsonValue]:
    trace_path.parent.mkdir(parents=True)
    receipts = [
        {
            "event_id": f"append-{index}",
            "seq": index,
            "scheduled_s": 10 + index * 0.08,
            "start_s": 10 + index * 0.08 + send_delay_s,
            "completed_s": 10 + index * 0.08 + send_delay_s + 0.001,
        }
        for index in range(4)
    ]
    packet_samples = output_samples or [1764] * len(output_times_s)
    assert len(packet_samples) == len(output_times_s)
    records = [
        {
            "direction": "receive",
            "time_s": timestamp_s,
            "event": {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x00\x00" * samples).decode("ascii"),
                **(
                    {"sglang": {"media_time": {"t_start_ms": index * 80}}}
                    if media_time
                    else {}
                ),
            },
        }
        for index, (timestamp_s, samples) in enumerate(
            zip(output_times_s, packet_samples)
        )
    ]
    records.extend(
        [
            {
                "direction": "receive",
                "time_s": 10.34,
                "event": {"type": "response.done", "response": {"status": "completed"}},
            },
            {
                "direction": "receive",
                "time_s": 10.35,
                "event": {"type": "sglang.input_audio.drained"},
            },
            {
                "direction": "receive",
                "time_s": 10.36,
                "event": {"type": "session.closed"},
            },
        ]
    )
    if failed:
        records.append(
            {
                "direction": "error",
                "time_s": 10.37,
                "event": {"message": "server disconnected"},
            }
        )
    trace_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
        json.dumps({"session_start_s": 10.0, "appends": receipts})
    )
    return session_metrics(
        trace_path,
        session_id=trace_path.parent.name,
        input_duration_s=INPUT_DURATION_S,
        profile=DEFAULT_PROFILE,
        reserve_s=0.08,
    )


def test_synthetic_serving_metrics(tmp_path: Path) -> None:
    assert distribution([0, 1, 2, 3])["p75"] == pytest.approx(2.25)
    assert distribution([])["p75"] is None
    perfect = recorded_session(
        tmp_path / "perfect" / "trace.jsonl", [10, 10.08, 10.16, 10.24]
    )
    assert perfect["success"] is True
    assert perfect["ttfa_s"] == pytest.approx(0)
    assert perfect["output_gap_s"]["p99"] == pytest.approx(0.08)
    assert perfect["output_gap_s"]["p75"] == pytest.approx(0.08)
    assert perfect["output_gap_excess_s"]["max"] == pytest.approx(0)
    assert perfect["output_drift_s"]["max"] == pytest.approx(0)
    assert perfect["final_output_drift_s"] == pytest.approx(0)
    assert perfect["required_playout_buffer_s"] == pytest.approx(0)
    assert perfect["late_send_rate"] == 0
    assert perfect["output_coverage"] == pytest.approx(1)
    assert perfect["underrun_count"] == 0
    assert perfect["underrun_ratio"] == 0

    stalled = recorded_session(
        tmp_path / "stalled" / "trace.jsonl", [10, 10.08, 10.28, 10.36]
    )
    assert stalled["output_gap_s"]["max"] == pytest.approx(0.2)
    assert stalled["output_gap_excess_s"]["max"] == pytest.approx(0.12)
    assert stalled["output_drift_s"]["p75"] == pytest.approx(0.12)
    assert stalled["final_output_drift_s"] == pytest.approx(0.12)
    assert stalled["required_playout_buffer_s"] == pytest.approx(0.12)
    assert stalled["underrun_count"] == 1
    assert stalled["underrun_total_s"] == pytest.approx(0.04)
    assert stalled["underrun_ratio"] == pytest.approx(0.125)

    sparse = recorded_session(tmp_path / "sparse" / "trace.jsonl", [10, 10.08])
    assert sparse["output_coverage"] == pytest.approx(0.5)
    assert sparse["underrun_total_s"] == pytest.approx(0.16)
    assert sparse["underrun_ratio"] == pytest.approx(0.5)

    late = recorded_session(
        tmp_path / "late" / "trace.jsonl",
        [10, 10.08, 10.16, 10.24],
        send_delay_s=0.05,
    )
    assert late["send_lateness_s"]["p99"] == pytest.approx(0.05)
    assert late["late_send_count"] == 4
    assert late["late_send_rate"] == 1

    aligned = recorded_session(
        tmp_path / "aligned" / "trace.jsonl",
        [10.1, 10.18, 10.26, 10.34],
        send_delay_s=0.02,
        media_time=True,
    )
    assert aligned["media_schedule_lag_s"]["n"] == 4
    assert aligned["media_schedule_lag_s"]["p95"] == pytest.approx(0.1)
    assert aligned["media_send_lag_s"]["p95"] == pytest.approx(0.08)

    batched = recorded_session(
        tmp_path / "batched" / "trace.jsonl",
        [10, 10.1, 10.19],
        output_samples=[3528, 1764, 1764],
    )
    assert batched["output_gap_excess_s"]["max"] == pytest.approx(0.01)
    assert batched["final_output_drift_s"] == pytest.approx(-0.05)

    silent = recorded_session(tmp_path / "silent" / "trace.jsonl", [])
    assert silent["success"] is False
    assert silent["ttfa_s"] is None
    assert silent["output_drift_s"]["n"] == 0
    assert silent["final_output_drift_s"] is None
    assert silent["required_playout_buffer_s"] is None
    assert silent["output_coverage"] == 0
    assert silent["underrun_total_s"] == INPUT_DURATION_S
    assert silent["underrun_ratio"] == 1

    failed = recorded_session(
        tmp_path / "failed" / "trace.jsonl", [10, 10.08], failed=True
    )
    aggregate = aggregate_sessions([perfect, failed, silent])
    assert aggregate["attempted_sessions"] == 3
    assert aggregate["successful_sessions"] == 1
    assert aggregate["output_coverage"]["p50"] == pytest.approx(0.5)
    assert aggregate["ttfa_s"]["n"] == 2
    assert aggregate["final_output_drift_s"]["n"] == 2
    assert aggregate["required_playout_buffer_s"]["p95"] == pytest.approx(0)
    assert aggregate["media_schedule_lag_s"]["n"] == 0
    assert aggregate["late_send_rate"] == 0
    assert aggregate["underrun_ratio"] == pytest.approx(0.5)


def test_scheduled_deadlines_do_not_follow_late_sends() -> None:
    assert [scheduled_send_s(10, index) for index in range(4)] == pytest.approx(
        [10, 10.08, 10.16, 10.24]
    )
    assert scheduled_send_s(10, 3) == pytest.approx(10.24)


def test_late_start_catches_up_to_absolute_schedule(tmp_path: Path) -> None:
    async def run() -> list[dict[str, JsonValue]]:
        peer = DuplexPeer()
        async with websockets.serve(peer.handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            gate = asyncio.get_running_loop().create_future()
            gate.set_result(time.perf_counter() - 0.25)
            trace_path = tmp_path / "trace.jsonl"
            await run_session(
                f"ws://127.0.0.1:{port}/v1/realtime",
                b"\x00\x00" * (12 * PACKET_BYTES // 2),
                scenario="continuous",
                trace_path=trace_path,
                start_gate=gate,
            )
            return json.loads(trace_path.with_name(SEND_RECEIPTS_FILE).read_text())[
                "appends"
            ]

    appends = asyncio.run(run())
    assert len(appends) == 12
    assert appends[0]["start_s"] - appends[0]["scheduled_s"] >= 0.25
    assert [r["scheduled_s"] - appends[0]["scheduled_s"] for r in appends] == (
        pytest.approx([index * 0.08 for index in range(12)])
    )
    assert appends[2]["start_s"] - appends[0]["start_s"] < 0.08
    assert appends[-1]["start_s"] - appends[-1]["scheduled_s"] < 0.04


def test_delayed_iteration_preserves_later_deadlines(tmp_path: Path) -> None:
    original_send = websockets.ClientConnection.send

    async def delayed_send(
        connection: websockets.ClientConnection, message: str
    ) -> None:
        event = json.loads(message)
        if event["type"] == "input_audio_buffer.append" and event["sglang"]["seq"] == 1:
            await asyncio.sleep(0.24)
        else:
            pass
        await original_send(connection, message)

    async def run() -> dict[str, JsonValue]:
        async with websockets.serve(DuplexPeer().handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            with patch.object(websockets.ClientConnection, "send", delayed_send):
                await run_session(
                    f"ws://127.0.0.1:{port}/v1/realtime",
                    b"\x00\x00" * (6 * PACKET_BYTES // 2),
                    scenario="continuous",
                    trace_path=tmp_path / "trace.jsonl",
                )
        return json.loads((tmp_path / SEND_RECEIPTS_FILE).read_text())

    receipts = asyncio.run(run())
    appends = receipts["appends"]
    assert [r["scheduled_s"] for r in appends] == pytest.approx(
        [receipts["session_start_s"] + index * 0.08 for index in range(6)]
    )
    assert appends[2]["start_s"] - appends[2]["scheduled_s"] >= 0.15
    assert appends[3]["start_s"] - appends[2]["start_s"] < 0.04


@pytest.mark.parametrize("profile", list(PROFILES))
def test_capacity_two_admission(tmp_path: Path, profile: ProfileName) -> None:
    async def run() -> dict[str, JsonValue]:
        active = 0

        def admit(connection: ServerConnection, request: Request) -> Response | None:
            nonlocal active
            if active >= 2:
                return connection.respond(503, "connection capacity exhausted")
            else:
                active += 1
                return None

        async def handler(websocket: ServerConnection) -> None:
            nonlocal active
            try:
                await DuplexPeer().handler(websocket)
            finally:
                active -= 1

        async with websockets.serve(
            handler, "127.0.0.1", 0, process_request=admit
        ) as server:
            port = server.sockets[0].getsockname()[1]
            summary = await run_concurrency(
                f"ws://127.0.0.1:{port}/v1/realtime",
                b"\x00\x00" * (4 * PACKET_BYTES // 2),
                concurrency=4,
                profile=profile,
                output_dir=tmp_path / "c4",
                timeout_s=5,
                reserve_s=0.08,
            )
            assert active == 0
            next_level = await run_concurrency(
                f"ws://127.0.0.1:{port}/v1/realtime",
                b"\x00\x00" * (4 * PACKET_BYTES // 2),
                concurrency=2,
                profile=profile,
                output_dir=tmp_path / "c2",
                timeout_s=5,
                reserve_s=0.08,
            )
            assert next_level["aggregate"]["admitted_sessions"] == 2
            return summary

    summary = asyncio.run(run())
    aggregate = summary["aggregate"]
    assert aggregate["attempted_sessions"] == 4
    assert aggregate["admitted_sessions"] == 2
    assert aggregate["rejected_sessions"] == 2
    assert aggregate["successful_admitted_sessions"] == 2
    assert aggregate["output_coverage"]["n"] == (
        2 if PROFILES[profile].continuous_output else 0
    )
    print_summary(summary)
    for session in summary["sessions"]:
        assert Path(session["trace_file"]).exists()
        assert Path(session["receipts_file"]).exists()
        if session["status"] == "rejected":
            assert session["admitted"] is False
            assert session["errors"] == []
            trace = Path(session["trace_file"]).read_text()
            assert '"exhausted": true' in trace
        else:
            assert session["success"] is True


@pytest.mark.parametrize("profile", list(PROFILES))
def test_native_unit_and_response_metrics(tmp_path: Path, profile: ProfileName) -> None:
    native_s = PROFILES[profile].native_unit_ms / 1000
    input_duration_s = 2.2 * native_s
    trace_path = tmp_path / "trace.jsonl"
    events = [
        (10.0, {"type": "session.created"}),
        (10.0, {"type": "response.created", "response": {"id": "r0"}}),
        (
            10.01,
            {
                "type": "response.output_text.delta",
                "response_id": "r0",
                "delta": "hello",
            },
        ),
        (10 + native_s + 0.03, {"type": "sglang.unit.done", "unit_id": "unit_0"}),
        (10 + 2 * native_s + 0.07, {"type": "sglang.unit.done", "unit_id": "unit_1"}),
        (
            10 + input_duration_s + 0.05,
            {"type": "sglang.unit.done", "unit_id": "unit_2"},
        ),
        (
            11.0,
            {"type": "response.done", "response": {"id": "r0", "status": "completed"}},
        ),
        (12.0, {"type": "response.created", "response": {"id": "r1"}}),
        (
            13.0,
            {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
        ),
        (14.0, {"type": "sglang.input_audio.drained"}),
        (14.1, {"type": "session.closed"}),
    ]
    for response_id, timestamp_s, duration_s in (
        ("r0", 10.1, 0.2),
        ("r0", 10.35, 0.3),
        ("r0", 10.7, 0.1),
        ("r1", 12.2, 0.1),
        ("r1", 12.3, 0.1),
    ):
        samples = round(duration_s * PROFILES[profile].output_sample_rate)
        events.append(
            (
                timestamp_s,
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(b"\x00\x00" * samples).decode("ascii"),
                },
            )
        )
    trace_path.write_text(
        "".join(
            json.dumps({"direction": "receive", "time_s": timestamp_s, "event": event})
            + "\n"
            for timestamp_s, event in sorted(
                events, key=lambda observation: observation[0]
            )
        )
    )
    append_count = round(input_duration_s / 0.08 + 0.499999)
    trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
        json.dumps(
            {
                "session_start_s": 10.0,
                "appends": [
                    {
                        "seq": index,
                        "scheduled_s": 10 + index * 0.08,
                        "start_s": 10 + index * 0.08,
                    }
                    for index in range(append_count)
                ],
            }
        )
    )
    metrics = session_metrics(
        trace_path,
        session_id="s0",
        input_duration_s=input_duration_s,
        profile=profile,
        reserve_s=0.08,
    )
    assert metrics["success"] is True
    assert metrics["unit_lag_values_s"] == pytest.approx([0.03, 0.07])
    assert metrics["excluded_terminal_units"] == ["unit_2"]
    assert metrics["response_ttfa_values_s"] == pytest.approx([0.1, 0.2])
    assert metrics["response_gap_excess_values_s"] == pytest.approx([0.05, 0.05, 0])
    assert metrics["response_max_drift_values_s"] == pytest.approx([0.1, 0])
    assert metrics["response_required_playout_buffer_values_s"] == pytest.approx(
        [0.1, 0]
    )
    if not PROFILES[profile].continuous_output:
        failed = empty_session_metrics(
            tmp_path / "missing.jsonl", "failed", input_duration_s, profile
        )
        failed["admitted"] = True
        aggregate = aggregate_sessions([metrics, failed])
        for name in (
            "ttfa_s",
            "session_ttfa_s",
            "output_gap_s",
            "output_gap_excess_s",
            "output_drift_s",
            "output_coverage",
            "final_output_drift_s",
            "required_playout_buffer_s",
            "underrun_count",
            "underrun_total_s",
            "underrun_ratio",
        ):
            assert metrics[name] is None
            assert failed[name] is None
            assert aggregate[name] is None or aggregate[name]["n"] == 0
    else:
        assert metrics["ttfa_s"] == pytest.approx(0.1)


def test_send_lateness_invalidates_run(tmp_path: Path) -> None:
    original_metrics = session_metrics

    def late_metrics(
        trace_path: Path,
        *,
        session_id: str,
        input_duration_s: float,
        profile: ProfileName,
        reserve_s: float,
    ) -> dict[str, JsonValue]:
        metrics = original_metrics(
            trace_path,
            session_id=session_id,
            input_duration_s=input_duration_s,
            profile=profile,
            reserve_s=reserve_s,
        )
        metrics["send_lateness_s"] = distribution([0.05])
        return metrics

    async def run() -> dict[str, JsonValue]:
        async with websockets.serve(DuplexPeer().handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            with patch(
                "benchmarks.duplex.serving.session_metrics", side_effect=late_metrics
            ):
                return await run_concurrency(
                    f"ws://127.0.0.1:{port}/v1/realtime",
                    b"\x00\x00" * (4 * PACKET_BYTES // 2),
                    concurrency=1,
                    profile=DEFAULT_PROFILE,
                    output_dir=tmp_path / "late",
                    timeout_s=5,
                    reserve_s=0.08,
                    loop_lag_limit_s=1.0,
                )

    summary = asyncio.run(run())
    assert summary["loop_lag_s"]["p99"] < summary["loop_lag_limit_s"]
    assert summary["client_timing_valid"] is False


def test_coordinator_keeps_failed_session(tmp_path: Path) -> None:
    async def run() -> dict[str, JsonValue]:
        peers: list[DuplexPeer] = []

        async def handler(websocket: ServerConnection) -> None:
            peer = DuplexPeer("close_without_update" if len(peers) == 0 else "healthy")
            peers.append(peer)
            await peer.handler(websocket)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            result = await run_concurrency(
                f"ws://127.0.0.1:{port}/v1/realtime",
                b"\x00\x00" * (4 * PACKET_BYTES // 2),
                concurrency=3,
                profile=DEFAULT_PROFILE,
                output_dir=tmp_path / "c3",
                timeout_s=5,
                reserve_s=0.08,
                loop_lag_limit_s=1e-9,
            )
        assert len(peers) == 3
        return result

    summary = asyncio.run(run())
    assert summary["configured_sessions"] == 2
    assert summary["aggregate"]["attempted_sessions"] == 3
    assert summary["aggregate"]["successful_sessions"] == 2
    assert summary["client_timing_valid"] is False
    assert summary["loop_lag_s"]["n"] > 0
    assert summary["aggregate"]["underrun_ratio"] >= 1 / 3
    assert [session["success"] for session in summary["sessions"]] == [
        False,
        True,
        True,
    ]
    for session in summary["sessions"]:
        assert Path(session["trace_file"]).exists()
        receipts = json.loads(Path(session["receipts_file"]).read_text())
        if session["success"]:
            assert receipts["session_start_s"] == summary["common_start_s"]
        else:
            assert receipts["session_start_s"] is None


def test_serving_cli_sweep_against_fake_server(tmp_path: Path) -> None:
    audio_path = tmp_path / "input.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"\x00\x00" * (4 * PACKET_BYTES // 2))

    async def run() -> tuple[int, str]:
        async def handler(websocket: ServerConnection) -> None:
            await DuplexPeer().handler(websocket)

        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "benchmarks.duplex.serving",
                "--url",
                f"ws://127.0.0.1:{port}/v1/realtime",
                "--audio",
                str(audio_path),
                "--profile",
                DEFAULT_PROFILE,
                "--concurrencies",
                "1,2",
                "--repeats",
                "2",
                "--stagger-ms",
                "80",
                "--output-dir",
                str(tmp_path / "results"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            assert process.returncode == 0, stderr.decode()
            return process.returncode, stdout.decode()

    _, output = asyncio.run(run())
    assert "2/2" in output
    assert "4/4" in output
    assert "late sends" in output
    assert "underrun sessions" in output
    assert "underrun ratio" in output
    top_level = json.loads((tmp_path / "results" / "summary.json").read_text())
    summaries = top_level["runs"]
    assert [run["aggregate"]["attempted_sessions"] for run in summaries] == [1, 2, 1, 2]
    assert [run["aggregate"]["successful_sessions"] for run in summaries] == [
        1,
        2,
        1,
        2,
    ]
    assert (tmp_path / "results" / "warmup" / "summary.json").exists()
    assert [level["aggregate"]["ttfa_s"]["n"] for level in top_level["levels"]] == [
        2,
        4,
    ]
    assert summaries[1]["sessions"][1]["ttfa_s"] is not None
    starts = [
        json.loads(Path(session["receipts_file"]).read_text())["session_start_s"]
        for session in summaries[1]["sessions"]
    ]
    assert starts[1] - starts[0] == pytest.approx(0.04)
