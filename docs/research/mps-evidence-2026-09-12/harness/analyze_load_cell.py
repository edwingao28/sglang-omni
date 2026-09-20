"""Descriptive analysis of one saved finite-cohort load cell. CPU/local only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    if not values:
        return None
    values = sorted(values)

    def percentile(q):
        at = (len(values) - 1) * q
        lo, hi = math.floor(at), math.ceil(at)
        return values[lo] + (values[hi] - values[lo]) * (at - lo)

    return dict(n=len(values), min=values[0], mean=statistics.mean(values),
                p50=percentile(.5), p95=percentile(.95), max=values[-1])


def outstanding(intervals, start, end, checkpoints):
    """Half-open intervals; group simultaneous starts/ends without false peaks."""
    events = Counter()
    for left, right in intervals:
        events[left] += 1
        events[right] -= 1
    count = sum(change for at, change in events.items() if at <= start)
    peak, area, previous = count, 0, start
    for at in sorted(at for at in events if start < at < end):
        area += count * (at - previous)
        count += events[at]
        peak, previous = max(peak, count), at
    area += count * (end - previous)
    return dict(peak=peak, time_mean=area / (end - start) if end > start else None,
                checkpoints={name: sum(left <= at < right for left, right in intervals)
                             for name, at in checkpoints.items()})


def arrival_analysis(arrivals, speed):
    rows, metrics = arrivals.get("per_request", []), speed.get("per_request", [])
    expected = arrivals["expected_samples"]
    result = dict(expected=expected, arrival_rows=len(rows), metric_rows=len(metrics),
                  status=arrivals.get("status"), requested_rate=arrivals.get("requested_rate"),
                  concurrency=arrivals.get("concurrency"), issues=[])
    if arrivals.get("status") != "collected" or arrivals.get("concurrency") != 0:
        result["issues"].append("capture incomplete or not open-loop C0")
    schedule = arrivals["schedule"]
    schedule_hash = hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
    if schedule_hash != arrivals.get("schedule_sha256"):
        result["issues"].append("schedule hash mismatch")
    if len(rows) != expected or len(metrics) != expected:
        result["issues"].append("incomplete expected population")
    offsets = schedule.get("offsets_ns", [])
    if (len(offsets) != expected or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets != sorted(offsets)):
        result["issues"].append("invalid sealed schedule length or offsets")
        return result
    if [row.get("nominal_offset_ns") for row in rows] != offsets:
        result["issues"].append("row offsets differ from sealed schedule")
        return result
    by_id = {row["id"]: row for row in metrics}
    if [r["sample_id"] for r in rows] != schedule["sample_ids"] or len(by_id) != len(metrics):
        result["issues"].append("sample identity/order mismatch")
    required = ("nominal_arrival_ns", "nominal_arrival_monotonic_ns", "dispatch_monotonic_ns",
                "dispatch_ns", "task_start_ns", "send_ns", "end_ns", "dispatch_lag_ns")
    if not rows or any(any(type(r.get(k)) is not int for k in required) for r in rows):
        result["issues"].append("missing timestamps; full-cohort rate/drain/outstanding unavailable")
        return result
    for row in rows:
        metric = by_id.get(row["sample_id"], {})
        if any(row.get(a) != metric.get(b) for a, b in
               (("send_ns", "request_start_ns"), ("end_ns", "request_end_ns"),
                ("server_request_id", "server_request_id"), ("worker_id", "worker_id"), ("is_success", "is_success"))):
            result["issues"].append("arrival/metric mismatch: " + row["sample_id"])
        if (row.get("nominal_arrival_ns") != arrivals["origin_wall_ns"] + row["nominal_offset_ns"]
                or row.get("nominal_arrival_monotonic_ns") != arrivals["origin_monotonic_ns"] + row["nominal_offset_ns"]
                or row.get("dispatch_lag_ns") != row["dispatch_monotonic_ns"] - row["nominal_arrival_monotonic_ns"]):
            result["issues"].append("nominal/dispatch clock identity mismatch: " + row["sample_id"])
    if any(not r["nominal_arrival_ns"] <= r["dispatch_ns"] <= r["task_start_ns"] <= r["send_ns"] <= r["end_ns"] for r in rows):
        result["issues"].append("nominal/dispatch/task/send/end clock ordering mismatch")
        return result
    origin = arrivals["origin_wall_ns"]
    first_send, last_send = min(r["send_ns"] for r in rows), max(r["send_ns"] for r in rows)
    last_end = max(r["end_ns"] for r in rows)
    nominal_end = origin + schedule["offsets_ns"][-1]
    last_dispatch = max(r["dispatch_ns"] for r in rows)
    completed = sum(r.get("is_success") is True for r in rows)
    checkpoints = dict(last_nominal_arrival=nominal_end, last_send=last_send)
    rate = lambda count, ns: count * 1e9 / ns if ns > 0 else None
    result.update(
        successful=completed, failed=len(rows) - completed,
        schedule_sha256=schedule_hash,
        origin_wall_ns=origin, first_send_ns=first_send, last_send_ns=last_send, last_end_ns=last_end,
        last_nominal_arrival_ns=nominal_end,
        nominal_arrival_span_s=(nominal_end - origin) / 1e9,
        send_envelope_s=(last_send - first_send) / 1e9,
        request_envelope_s=(last_end - first_send) / 1e9,
        cohort_origin_to_end_s=(last_end - origin) / 1e9,
        nominal_drain_s=(last_end - nominal_end) / 1e9,
        last_send_to_last_completion_s=(last_end - last_send) / 1e9,
        realized_scheduled_rate_n_over_origin_to_last=rate(expected, nominal_end - origin),
        dispatched_rate_n_over_origin_to_last=rate(len(rows), last_dispatch - origin),
        sent_rate_n_over_origin_to_last=rate(len(rows), last_send - origin),
        sent_interarrival_rate_n_minus_1=rate(len(rows) - 1, last_send - first_send),
        dispatch_lag_ms=distribution([r["dispatch_lag_ns"] / 1e6 for r in rows]),
        nominal_to_send_ms=distribution([(r["send_ns"] - r["nominal_arrival_ns"]) / 1e6 for r in rows]),
        dispatch_to_task_ms=distribution([(r["task_start_ns"] - r["dispatch_ns"]) / 1e6 for r in rows]),
        task_start_to_send_ms=distribution([(r["send_ns"] - r["task_start_ns"]) / 1e6 for r in rows]),
        http_latency_s=distribution([(r["end_ns"] - r["send_ns"]) / 1e9 for r in rows]),
        nominal_to_completion_s=distribution([(r["end_ns"] - r["nominal_arrival_ns"]) / 1e9 for r in rows]),
        http_inflight=outstanding([(r["send_ns"], r["end_ns"]) for r in rows], origin, last_end, checkpoints),
        due_but_not_sent=outstanding([(r["nominal_arrival_ns"], r["send_ns"]) for r in rows], origin, last_end, checkpoints),
        scheduled_unfinished=outstanding([(r["nominal_arrival_ns"], r["end_ns"]) for r in rows], origin, last_end, checkpoints),
        worker_counts=dict(Counter(r.get("worker_id") for r in rows)),
    )
    result["rate_scope"] = "Finite realization: N/(last-origin) includes first scheduled wait; (N-1)/(last-first) describes send interarrivals. Neither is service capacity or a replacement for saved runner throughput."
    return result


def process_role(row, rows, roots, mps):
    pid = row["pid"]
    if pid == mps.get("daemon_pid"):
        return "mps-daemon"
    if pid in {r["server_pid"] for r in mps.get("clients", [])}:
        return "mps-server"
    seen = set()
    while pid not in seen:
        if pid in roots:
            return roots[pid]
        seen.add(pid)
        if pid not in rows:
            break
        pid = rows[pid]["ppid"]
    return "owned-other"


def cpu_analysis(samples, ready, run, start, end):
    roots = {p["pid"]: p["name"] for p in run["processes"]}
    roots.update({int(pid): "cell-controller" for pid in ready.get("root_start_ticks", {})})
    mps = ready.get("mps_ownership") or {}
    mps_clients = {r["client_pid"] for r in mps.get("clients", [])}
    capacities = {f"replica-{i}": set(cpus) for i, cpus in enumerate(run["cpu_sets"])}
    capacities.update(router=set(run["router_cpu_set"]), client=set(run["client_cpu_set"]))
    required = {int(pid): ticks for pid, ticks in ready.get("root_start_ticks", {}).items()}
    required.update({int(pid): ticks for pid, ticks in mps.get("process_start_ticks", {}).items()})
    processes, intervals, issues = {}, [], []
    edge_intervals = []
    for a, b in zip(samples, samples[1:]):
        left, right = a["timestamp_ns"], b["timestamp_ns"]
        if right <= start or left >= end:
            continue
        if not start <= left < right <= end:
            edge_intervals.append(dict(start_ns=left, end_ns=right))
            continue
        hz = a["clock_ticks_per_second"]
        if hz != b["clock_ticks_per_second"] or hz <= 0:
            issues.append("clock tick rate mismatch")
            continue
        previous, current = ({p["pid"]: p for p in s["processes"]} for s in (a, b))
        dt = (right - left) / 1e9
        interval = dict(start_ns=left, end_ns=right, wall_s=dt, cpu_s=0., roles_cpu_s={},
                        unmatched_pids=sorted(previous.keys() ^ current.keys()), identity_changes=[],
                        missing_required_pids=sorted(pid for pid, ticks in required.items()
                            if previous.get(pid, {}).get("start_ticks") != ticks or current.get(pid, {}).get("start_ticks") != ticks))
        roles_cpu = defaultdict(float)
        for pid in previous.keys() & current.keys():
            p, q = previous[pid], current[pid]
            if p["start_ticks"] != q["start_ticks"]:
                interval["identity_changes"].append(pid)
                continue
            cpu = (q["user_ticks"] + q["system_ticks"] - p["user_ticks"] - p["system_ticks"]) / hz
            if cpu < 0:
                issues.append(f"negative CPU counter delta: {pid}")
                continue
            role = process_role(q, {**previous, **current}, roots, mps)
            key = f'{pid}:{q["start_ticks"]}'
            entry = processes.setdefault(key, dict(pid=pid, start_ticks=q["start_ticks"], name=q["name"], role=role,
                is_mps_client=pid in mps_clients, cpu_s=0., covered_wall_s=0., interval_one_core_pct=[],
                cpu_affinity=set(), threads=[], affinity_changed=False))
            entry["cpu_s"] += cpu
            entry["covered_wall_s"] += dt
            entry["interval_one_core_pct"].append(100 * cpu / dt)
            entry["cpu_affinity"].update(p["cpu_affinity"] + q["cpu_affinity"])
            entry["threads"].extend([p["threads"], q["threads"]])
            entry["affinity_changed"] |= p["cpu_affinity"] != q["cpu_affinity"]
            roles_cpu[role] += cpu
        interval["roles_cpu_s"] = dict(roles_cpu)
        interval["cpu_s"] = sum(roles_cpu.values())
        intervals.append(interval)
    covered = sum(i["wall_s"] for i in intervals)
    roles = []
    for role in sorted({p["role"] for p in processes.values()}):
        members = [p for p in processes.values() if p["role"] == role]
        cpu = sum(p["cpu_s"] for p in members)
        affinity = set().union(*(p["cpu_affinity"] for p in members))
        declared = capacities.get(role)
        one_core_pct = 100 * cpu / covered if covered else None
        roles.append(dict(role=role, pids=sorted(p["pid"] for p in members), observed_cpu_s=cpu,
            mean_one_core_pct_over_covered_window=one_core_pct, cpu_affinity_union=sorted(affinity),
            declared_cpu_set=sorted(declared) if declared else None,
            mean_declared_capacity_pct=one_core_pct / len(declared) if declared and affinity <= declared and covered else None))
    for p in processes.values():
        p["cpu_affinity"] = sorted(p["cpu_affinity"])
        p["threads"] = dict(min=min(p["threads"]), max=max(p["threads"]))
        p["one_core_pct"] = distribution(p.pop("interval_one_core_pct"))
        p["mean_one_core_pct"] = 100 * p["cpu_s"] / p["covered_wall_s"]
        p["fraction_of_covered_window_observed"] = p["covered_wall_s"] / covered
    return dict(window_start_ns=start, window_end_ns=end, window_s=(end - start) / 1e9,
        sample_count=len(samples), complete_inside_intervals=len(intervals), covered_wall_s=covered,
        covered_window_fraction=covered * 1e9 / (end - start), edge_intervals_not_apportioned=edge_intervals,
        sample_spacing_s=distribution([(b["timestamp_ns"] - a["timestamp_ns"]) / 1e9 for a, b in zip(samples, samples[1:])]),
        sampling_brackets_window=bool(samples and samples[0]["timestamp_ns"] <= start and samples[-1]["timestamp_ns"] >= end),
        readiness_before_window=ready.get("first_sample_timestamp_ns", end + 1) <= start,
        mps_ownership_verified=bool(mps.get("verified_owned")), mps_ownership=mps,
        total_observed_cpu_s=sum(i["cpu_s"] for i in intervals),
        aggregate_mean_one_core_pct=100 * sum(i["cpu_s"] for i in intervals) / covered if covered else None,
        process_intervals_with_missing_required=sum(bool(i["missing_required_pids"]) for i in intervals),
        process_intervals_with_unmatched=sum(bool(i["unmatched_pids"]) for i in intervals),
        roles=roles, processes=list(processes.values()), intervals=intervals, issues=issues,
        scope="All-thread process CPU deltas in complete adjacent sample intervals only. Window interval coverage is not complete process CPU coverage: missing/PID-reused contributions are unknown, not zero. Affinity snapshots describe process leaders, not all threads. Per-role percentages divide observed CPU by declared budgets; shared affinities cannot be added as separate physical capacity. No hot-thread/GIL inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = {}

    def read(relative):
        path = args.run / relative
        raw = path.read_bytes()
        sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines()] if path.suffix == ".jsonl" else json.loads(raw)

    run, speed, arrivals = read("run.json"), read("metrics/speed_results.json"), read("metrics/arrivals.json")
    arrival = arrival_analysis(arrivals, speed)
    result = dict(run_id=run["run_id"], run_status=run["status"], run_metadata=run,
                  source_sha256=sources, arrival=arrival, upstream_speed_summary=speed["summary"],
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  quality_status="available; review separately" if (args.run / "metrics/wer_results.json").exists() else "pending",
                  scope="One finite 128-input load cell. No steady-state capacity, saturation, SM contention or production SLO claim.")
    if "first_send_ns" in arrival:
        ready = read("cpu-sampler-ready.json")
        ownership = read("mps-cpu-ownership.json") if ready.get("mps_enabled") else None
        result["receipt_issues"] = []
        if ready != run.get("cpu_sampler_ready") or ownership != ready.get("mps_ownership"):
            result["receipt_issues"].append("CPU readiness/ownership receipts differ")
        if speed["summary"] != run.get("metrics_summary") or arrival["schedule_sha256"] != run.get("arrival_schedule_sha256"):
            result["receipt_issues"].append("run metrics/schedule receipts differ")
        result["cpu"] = cpu_analysis(read("cpu-samples.jsonl"), ready, run,
                                     arrival["first_send_ns"], arrival["last_end_ns"])
        result["measurement_window_utc"] = [datetime.fromtimestamp(arrival[k] / 1e9, timezone.utc).isoformat()
                                             for k in ("first_send_ns", "last_end_ns")]
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(run_id=result["run_id"], output=str(args.output), arrival_issues=arrival["issues"])))


if __name__ == "__main__":
    main()
