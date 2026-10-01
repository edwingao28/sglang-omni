# SPDX-License-Identifier: Apache-2.0
"""Aggregate admission counts and applicable native serving measurements."""

from pydantic import JsonValue

from benchmarks.duplex.serving_metrics import LATE_SEND_THRESHOLD_S, distribution


def aggregate_sessions(sessions: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    send_count = sum(len(s["send_lateness_values_s"]) for s in sessions)
    late_send_count = sum(s["late_send_count"] for s in sessions)
    admitted = [s for s in sessions if s["admitted"]]
    playout = [s for s in admitted if s["underrun_total_s"] is not None]
    underrun_total_s = sum(s["underrun_total_s"] for s in playout) if playout else None
    aggregate = {
        "attempted_sessions": len(sessions),
        "admitted_sessions": len(admitted),
        "rejected_sessions": sum(s["status"] == "rejected" for s in sessions),
        "successful_admitted_sessions": sum(bool(s["success"]) for s in admitted),
        "successful_sessions": sum(bool(s["success"]) for s in admitted),
        "ttfa_s": distribution(
            [s["ttfa_s"] for s in sessions if s["ttfa_s"] is not None]
        ),
        "send_lateness_s": distribution(
            [value for s in sessions for value in s["send_lateness_values_s"]]
        ),
        "late_send_threshold_s": LATE_SEND_THRESHOLD_S,
        "late_send_count": late_send_count,
        "late_send_rate": late_send_count / send_count if send_count else None,
        "output_gap_s": distribution(
            [value for s in sessions for value in s["output_gap_values_s"]]
        ),
        "output_gap_excess_s": distribution(
            [value for s in sessions for value in s["output_gap_excess_values_s"]]
        ),
        "output_drift_s": distribution(
            [value for s in sessions for value in s["output_drift_values_s"]]
        ),
        "final_output_drift_s": distribution(
            [
                s["final_output_drift_s"]
                for s in sessions
                if s["final_output_drift_s"] is not None
            ]
        ),
        "required_playout_buffer_s": distribution(
            [
                s["required_playout_buffer_s"]
                for s in sessions
                if s["required_playout_buffer_s"] is not None
            ]
        ),
        "media_schedule_lag_s": distribution(
            [value for s in sessions for value in s["media_schedule_lag_values_s"]]
        ),
        "media_send_lag_s": distribution(
            [value for s in sessions for value in s["media_send_lag_values_s"]]
        ),
        "output_coverage": distribution(
            [s["output_coverage"] for s in admitted if s["output_coverage"] is not None]
        ),
        "playout_sessions": len(playout),
        "underrun_sessions": (
            sum(bool(s["underrun_count"]) for s in playout) if playout else None
        ),
        "underrun_count": (
            sum(s["underrun_count"] for s in playout) if playout else None
        ),
        "underrun_total_s": underrun_total_s,
        "underrun_worst_s": max((s["underrun_worst_s"] for s in playout), default=None),
        "underrun_ratio": (
            underrun_total_s / sum(s["input_duration_s"] for s in playout)
            if playout
            else None
        ),
    }
    for name in (
        "unit_lag",
        "response_ttfa",
        "response_gap_excess",
        "response_max_drift",
        "response_required_playout_buffer",
    ):
        aggregate[f"{name}_s"] = distribution(
            [v for s in admitted for v in s[f"{name}_values_s"] if v is not None]
        )
    aggregate["session_ttfa_s"] = aggregate["ttfa_s"]
    return aggregate
