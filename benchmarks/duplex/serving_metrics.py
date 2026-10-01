# SPDX-License-Identifier: Apache-2.0
"""Compute client-observable serving metrics from native duplex recordings."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.client import PACKET_BYTES, SAMPLE_RATE, SEND_RECEIPTS_FILE
from benchmarks.duplex.profiles import PROFILES, ProfileName

PLAYBACK_EPSILON_S = 1e-9
LATE_SEND_THRESHOLD_S = 0.02


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "n": 0,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "n": len(ordered),
        "p50": percentile(0.5),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def playback_underruns(
    packets: list[tuple[float, float]], end_s: float, reserve_s: float
) -> tuple[int, float, float]:
    assert packets
    start_s = packets[0][0] + reserve_s
    buffer_s = 0.0
    previous_s = start_s
    count = 0
    total_s = 0.0
    worst_s = 0.0
    for arrival_s, duration_s in [*packets, (end_s, 0.0)]:
        if arrival_s > end_s:
            break
        if arrival_s > previous_s:
            deficit_s = max(0.0, arrival_s - previous_s - buffer_s)
            if deficit_s > PLAYBACK_EPSILON_S:
                count += 1
                total_s += deficit_s
                worst_s = max(worst_s, deficit_s)
            buffer_s = max(0.0, buffer_s - (arrival_s - previous_s))
            previous_s = arrival_s
        buffer_s += duration_s
    return count, total_s, worst_s


def empty_session_metrics(
    trace_path: Path, session_id: str, input_duration_s: float, profile: ProfileName
) -> dict[str, JsonValue]:
    continuous = PROFILES[profile].continuous_output
    return {
        "session_id": session_id,
        "input_duration_s": input_duration_s,
        "trace_file": str(trace_path),
        "receipts_file": str(trace_path.with_name(SEND_RECEIPTS_FILE)),
        "success": False,
        "admitted": False,
        "status": "failed",
        "errors": [],
        "ttfa_s": None,
        "session_ttfa_s": None,
        "unit_lag_s": distribution([]),
        "unit_lag_values_s": [],
        "excluded_terminal_units": [],
        "response_ttfa_s": distribution([]),
        "response_ttfa_values_s": [],
        "response_gap_excess_s": distribution([]),
        "response_gap_excess_values_s": [],
        "response_max_drift_s": distribution([]),
        "response_max_drift_values_s": [],
        "response_required_playout_buffer_s": distribution([]),
        "response_required_playout_buffer_values_s": [],
        "responses": [],
        "send_lateness_s": distribution([]),
        "late_send_count": 0,
        "late_send_rate": None,
        "output_gap_s": distribution([]) if continuous else None,
        "output_gap_excess_s": distribution([]) if continuous else None,
        "output_drift_s": distribution([]) if continuous else None,
        "final_output_drift_s": None,
        "required_playout_buffer_s": None,
        "media_schedule_lag_s": distribution([]),
        "media_send_lag_s": distribution([]),
        "send_lateness_values_s": [],
        "output_gap_values_s": [],
        "output_gap_excess_values_s": [],
        "output_drift_values_s": [],
        "media_schedule_lag_values_s": [],
        "media_send_lag_values_s": [],
        "output_samples": 0,
        "output_duration_s": 0.0,
        "output_coverage": 0.0 if continuous else None,
        "underrun_count": 1 if continuous else None,
        "underrun_total_s": input_duration_s if continuous else None,
        "underrun_worst_s": input_duration_s if continuous else None,
        "underrun_ratio": 1.0 if continuous else None,
    }


def session_metrics(
    trace_path: Path,
    *,
    session_id: str,
    input_duration_s: float,
    profile: ProfileName,
    reserve_s: float,
) -> dict[str, JsonValue]:
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    receipts = json.loads(
        trace_path.with_name(SEND_RECEIPTS_FILE).read_text(encoding="utf-8")
    )
    start_s = receipts["session_start_s"]
    appends = receipts["appends"]
    metrics = empty_session_metrics(trace_path, session_id, input_duration_s, profile)
    admitted = receipts.get("admitted", False) or any(
        record["direction"] == "receive"
        or (
            record["direction"] == "admission"
            and record["event"].get("type") == "connection_admitted"
        )
        for record in records
    )
    rejected = not admitted and any(
        record["direction"] == "admission"
        and record["event"].get("http_status") == 503
        and record["event"].get("exhausted")
        for record in records
    )
    metrics["admitted"] = admitted
    if rejected:
        metrics["status"] = "rejected"
        return metrics
    else:
        pass
    native_unit_s = PROFILES[profile].native_unit_ms / 1000
    full_units = math.floor((input_duration_s + PLAYBACK_EPSILON_S) / native_unit_s)
    unit_lags: dict[str, float] = {}
    terminal_units: list[str] = []
    response_created_s: dict[str, float] = {}
    response_audio: dict[str, list[tuple[float, float]]] = {}
    append_by_media_ms = {
        round(append["seq"] * PACKET_BYTES / (2 * SAMPLE_RATE) * 1000, 3): append
        for append in appends
    }
    audio_packets: list[tuple[float, float]] = []
    media_schedule_lag: list[float] = []
    media_send_lag: list[float] = []
    seen_media_ms: set[float] = set()
    event_types: list[str] = []
    errors: list[str] = []
    output_samples = 0
    for record in records:
        direction = record["direction"]
        event = record["event"]
        event_type = event.get("type")
        if direction == "error":
            errors.append(str(event.get("message", "client error")))
        elif direction == "receive":
            event_types.append(event_type)
            if event_type == "sglang.unit.done":
                unit_id = event.get("unit_id")
                match = re.fullmatch(r"unit_(\d+)", unit_id or "")
                if match is None:
                    errors.append(f"invalid completed unit id: {unit_id}")
                elif int(match[1]) >= full_units:
                    terminal_units.append(unit_id)
                elif start_s is not None:
                    unit_lags.setdefault(
                        unit_id,
                        record["time_s"]
                        - (start_s + (int(match[1]) + 1) * native_unit_s),
                    )
                else:
                    pass
            elif event_type == "response.created":
                response = event.get("response", {})
                response_id = response.get("id")
                if isinstance(response_id, str):
                    response_created_s.setdefault(response_id, record["time_s"])
                else:
                    errors.append("response.created missing response id")
            elif event_type == "error":
                errors.append(str(event.get("error", "server error")))
            elif event_type == "response.done":
                response = event.get("response")
                if (
                    not isinstance(response, dict)
                    or response.get("status") != "completed"
                ):
                    errors.append(f"response did not complete: {response}")
            elif event_type == "session.closed" and event.get("reason") not in (
                None,
                "client_closed",
            ):
                errors.append(f"unexpected session close: {event.get('reason')}")
            elif event_type == "response.output_audio.delta":
                try:
                    pcm = base64.b64decode(event["delta"], validate=True)
                    if not pcm or len(pcm) % 2:
                        raise ValueError("output audio is empty or not PCM16")
                    duration_s = len(pcm) / (2 * PROFILES[profile].output_sample_rate)
                    output_samples += len(pcm) // 2
                    audio_packets.append((record["time_s"], duration_s))
                    response_id = event.get("response_id")
                    if isinstance(response_id, str):
                        response_audio.setdefault(response_id, []).append(
                            (record["time_s"], duration_s)
                        )
                    else:
                        pass
                    extension = event.get("sglang")
                    media_time = (
                        extension.get("media_time")
                        if isinstance(extension, dict)
                        else None
                    )
                    if isinstance(media_time, dict) and isinstance(
                        media_time.get("t_start_ms"), (int, float)
                    ):
                        media_ms = round(media_time["t_start_ms"], 3)
                        append = append_by_media_ms.get(media_ms)
                        if append is not None and media_ms not in seen_media_ms:
                            seen_media_ms.add(media_ms)
                            media_schedule_lag.append(
                                record["time_s"] - append["scheduled_s"]
                            )
                            media_send_lag.append(record["time_s"] - append["start_s"])
                except (KeyError, ValueError, binascii.Error) as exc:
                    errors.append(f"invalid output audio: {exc}")
    expected_appends = math.ceil(input_duration_s * SAMPLE_RATE * 2 / PACKET_BYTES)
    if len(appends) != expected_appends:
        errors.append(f"sent {len(appends)}/{expected_appends} input frames")
    if "sglang.input_audio.drained" not in event_types:
        errors.append("input did not drain")
    if "session.closed" not in event_types:
        errors.append("session did not close")
    if audio_packets and "response.done" not in event_types:
        errors.append("output response did not complete")
    if not audio_packets and PROFILES[profile].continuous_output:
        errors.append("no output audio")
    if start_s is None:
        errors.append("session did not start")
    lateness = [max(0.0, r["start_s"] - r["scheduled_s"]) for r in appends]
    continuous = PROFILES[profile].continuous_output
    playout_packets = audio_packets if continuous else []
    gaps = [
        current[0] - previous[0]
        for previous, current in zip(playout_packets, playout_packets[1:])
    ]
    gap_excess = [
        max(0.0, current[0] - previous[0] - previous[1])
        for previous, current in zip(playout_packets, playout_packets[1:])
    ]
    output_drift: list[float] = []
    if playout_packets:
        ideal_arrival_s = playout_packets[0][0]
        for arrival_s, duration_s in playout_packets:
            output_drift.append(arrival_s - ideal_arrival_s)
            ideal_arrival_s += duration_s
    else:
        pass
    late_send_count = sum(value > LATE_SEND_THRESHOLD_S for value in lateness)
    underrun_count, underrun_total_s, underrun_worst_s = (
        playback_underruns(
            audio_packets, start_s + input_duration_s + reserve_s, reserve_s
        )
        if continuous
        and audio_packets
        and start_s is not None
        and audio_packets[0][0] < start_s + input_duration_s
        else (1, input_duration_s, input_duration_s)
    )
    metrics.update(
        {
            "success": not errors,
            "status": "succeeded" if not errors else "failed",
            "errors": errors,
            "ttfa_s": (
                audio_packets[0][0] - start_s
                if continuous and audio_packets and start_s is not None
                else None
            ),
            "send_lateness_s": distribution(lateness),
            "late_send_count": late_send_count,
            "late_send_rate": late_send_count / len(lateness) if lateness else None,
            "output_gap_s": distribution(gaps),
            "output_gap_excess_s": distribution(gap_excess),
            "output_drift_s": distribution(output_drift),
            "final_output_drift_s": output_drift[-1] if output_drift else None,
            "required_playout_buffer_s": (
                max(0.0, max(output_drift)) if output_drift else None
            ),
            "media_schedule_lag_s": distribution(media_schedule_lag),
            "media_send_lag_s": distribution(media_send_lag),
            "send_lateness_values_s": lateness,
            "output_gap_values_s": gaps,
            "output_gap_excess_values_s": gap_excess,
            "output_drift_values_s": output_drift,
            "media_schedule_lag_values_s": media_schedule_lag,
            "media_send_lag_values_s": media_send_lag,
            "output_samples": output_samples,
            "output_duration_s": output_samples / PROFILES[profile].output_sample_rate,
            "output_coverage": (
                output_samples / PROFILES[profile].output_sample_rate / input_duration_s
                if continuous
                else None
            ),
            "underrun_count": underrun_count,
            "underrun_total_s": underrun_total_s,
            "underrun_worst_s": underrun_worst_s,
            "underrun_ratio": underrun_total_s / input_duration_s,
        }
    )
    metrics["session_ttfa_s"] = metrics["ttfa_s"]
    response_timings = []
    for response_id, packets in response_audio.items():
        created_s = response_created_s.get(response_id)
        response_gaps = [
            max(0.0, current[0] - previous[0] - previous[1])
            for previous, current in zip(packets, packets[1:])
        ]
        ideal_s = packets[0][0]
        drifts = []
        for arrival_s, duration_s in packets:
            drifts.append(arrival_s - ideal_s)
            ideal_s += duration_s
        response_timings.append(
            {
                "response_id": response_id,
                "ttfa_s": packets[0][0] - created_s if created_s is not None else None,
                "gap_excess_values_s": response_gaps,
                "max_drift_s": max(drifts),
                "required_playout_buffer_s": max(0.0, max(drifts)),
            }
        )
    common_values = {
        "unit_lag": list(unit_lags.values()),
        "response_ttfa": [
            r["ttfa_s"] for r in response_timings if r["ttfa_s"] is not None
        ],
        "response_gap_excess": [
            v for r in response_timings for v in r["gap_excess_values_s"]
        ],
        "response_max_drift": [r["max_drift_s"] for r in response_timings],
        "response_required_playout_buffer": [
            r["required_playout_buffer_s"] for r in response_timings
        ],
    }
    for name, values in common_values.items():
        metrics[f"{name}_values_s"] = values
        metrics[f"{name}_s"] = distribution(values)
    metrics["excluded_terminal_units"] = terminal_units
    metrics["responses"] = response_timings
    if not PROFILES[profile].continuous_output:
        for name in (
            "ttfa_s",
            "session_ttfa_s",
            "output_gap_s",
            "output_gap_excess_s",
            "output_drift_s",
            "final_output_drift_s",
            "required_playout_buffer_s",
            "output_coverage",
            "underrun_count",
            "underrun_total_s",
            "underrun_worst_s",
            "underrun_ratio",
        ):
            metrics[name] = None
        for name in (
            "output_gap_values_s",
            "output_gap_excess_values_s",
            "output_drift_values_s",
        ):
            metrics[name] = []
    else:
        pass
    return metrics
