"""Fixed time-window completion flow and sent-cohort latency; no capacity verdict."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_load_cell import cpu_analysis, distribution, outstanding
from sustained_protocol import (BLOCK_SECONDS, INPUT_SHA, SEND_SECONDS, SETTLING_SECONDS,
    digest, duration_offsets, sha, write_new)


def analyze_capture(capture, manifest):
    rows, expected = capture['per_request'], manifest['occurrences']
    issues = []
    ids = [r['occurrence_id'] for r in expected]
    offsets = manifest['offsets_ns']
    if (manifest['phase'] != 'service' or manifest['input_sha256'] != INPUT_SHA
            or len(manifest['base_sample_ids']) != 128 or len(set(manifest['base_sample_ids'])) != 128):
        issues.append('Base cohort / service manifest mismatch')
    if len(set(ids)) != len(ids) or [r.get('occurrence_id') for r in rows] != ids:
        issues.append('Occurrence IDs missing, duplicated, or reordered')
    if (offsets != duration_offsets(manifest['requested_rate'], manifest['duration_s'], manifest['seed'])
            or [r.get('nominal_offset_ns') for r in rows] != offsets
            or capture['schedule_sha256'] != digest({'occurrence_ids': ids, 'offsets_ns': offsets})):
        issues.append('Seeded duration schedule / row offsets differ')
    if (capture['duration_s'] != SEND_SECONDS or capture['settling_s'] != SETTLING_SECONDS
            or capture['concurrency'] != 0 or capture['seed'] != 42):
        issues.append('Fixed protocol controls differ')
    for i, row in enumerate(expected):
        if row['base_index'] != i % 128 or row['base_id'] != manifest['base_sample_ids'][i % 128] or row['cycle'] != i // 128:
            issues.append('Cyclic base input mapping differs')
            break
    origin = capture['origin']['mono_ns']
    start, end = origin + SETTLING_SECONDS * 10**9, origin + SEND_SECONDS * 10**9
    # The client may dispatch an arrival scheduled before the cutoff up to dispatch_grace_s late; such
    # sends are legal but fall outside the [30,150) window by their actual send time (counted below).
    send_limit = end + round(float(capture.get('dispatch_grace_s') or 0) * 1e9)
    late_sends = 0
    for row in rows:
        if 'send_mono_ns' in row:
            times = [origin + row['nominal_offset_ns'], row['dispatch_mono_ns'],
                     row['task_start_mono_ns'], row['send_mono_ns'], row['terminal_mono_ns']]
            if times != sorted(times) or row['send_mono_ns'] >= send_limit or origin + row['nominal_offset_ns'] >= end:
                issues.append('Invalid nominal/dispatch/task/send/terminal order or send after cutoff')
            elif row['send_mono_ns'] >= end:
                late_sends += 1
            result = row.get('result')
            if result and result['request_id'] != row['occurrence_id']:
                issues.append('Upstream result ID differs from occurrence')
            if row['outcome'] == 'success' and (not result or not result['is_success'] or not result.get('wav_path')
                    or not result.get('server_request_id') or not result.get('worker_id')):
                issues.append('Successful result lacks actual WAV / server / worker identity')
        elif row['outcome'] not in ('not_dispatched', 'not_sent_before_cutoff', 'client_exception', 'cancelled'):
            issues.append('Unsent occurrence has unaccounted outcome')
    sent = [r for r in rows if 'send_mono_ns' in r]
    # A terminal outcome includes success, rejection, timeout, cancellation and errors.
    # This is client transaction occupancy, NOT a server queue or GPU-ready-work count.
    intervals = [(r['send_mono_ns'], r['terminal_mono_ns']) for r in sent]

    def window(left, right):
        inflow = [r for r in sent if left <= r['send_mono_ns'] < right]
        exits = [r for r in sent if left <= r['terminal_mono_ns'] < right]
        nominal = sum(left <= origin + r['nominal_offset_ns'] < right for r in rows)
        pending_left = sum(a < left <= b for a, b in intervals)
        pending_right = sum(a < right <= b for a, b in intervals)
        seconds = (right-left)/1e9
        outputs = Counter(r['outcome'] for r in exits)
        due_left = sum(origin+r['nominal_offset_ns'] < left and r.get('send_mono_ns', right+1) >= left for r in rows)
        due_right = sum(origin+r['nominal_offset_ns'] < right and r.get('send_mono_ns', right+1) >= right for r in rows)
        return dict(start_offset_s=(left-origin)/1e9, end_offset_s=(right-origin)/1e9,
            seconds=seconds, nominal_arrivals=nominal, actual_sends=len(inflow), terminals=len(exits),
            terminal_outcomes=dict(outputs), success_completions=outputs['success'],
            nominal_rate=nominal/seconds, actual_offered_rate=len(inflow)/seconds,
            successful_completion_rate=outputs['success']/seconds,
            all_terminal_rate=len(exits)/seconds, pending_start=pending_left, pending_end=pending_right,
            pending_change=pending_right-pending_left,
            conservation_residual=pending_right-pending_left-len(inflow)+len(exits),
            due_unsent_start=due_left, due_unsent_end=due_right,
            client_transaction_outstanding=outstanding(intervals, left, right, {}))

    cohort = [r for r in sent if start <= r['send_mono_ns'] < end]
    success = [r for r in cohort if r['outcome'] == 'success']
    blocks = [window(origin+s*10**9, origin+(s+BLOCK_SECONDS)*10**9)
              for s in range(SETTLING_SECONDS, SEND_SECONDS, BLOCK_SECONDS)]
    primary = window(start, end)
    if any(b['conservation_residual'] for b in [primary, *blocks]):
        issues.append('Inflow / terminal / pending conservation failed')
    if capture['status'] != 'collected':
        issues.append('Capture did not complete its planned dispatch/drain lifecycle')
    return dict(issues=sorted(set(issues)), status=capture['status'], window=primary, blocks=blocks,
        sends_after_nominal_cutoff_within_grace=late_sends, dispatch_grace_s=capture.get('dispatch_grace_s'),
        total_planned=len(rows), total_sent=len(sent), outcomes=dict(Counter(r['outcome'] for r in rows)),
        sent_window_cohort=dict(count=len(cohort), outcomes=dict(Counter(r['outcome'] for r in cohort)),
            success_count=len(success), success_latency_s=distribution([(r['terminal_mono_ns']-r['send_mono_ns'])/1e9 for r in success]),
            upstream_success_latency_s=distribution([r['result']['latency_s'] for r in success]),
            all_terminal_elapsed_s=distribution([(r['terminal_mono_ns']-r['send_mono_ns'])/1e9 for r in cohort]),
            terminal_after_window=sum(r['terminal_mono_ns'] >= end for r in cohort),
            occurrence_ids=[r['occurrence_id'] for r in cohort]),
        drain_s=max(0, (max((r['terminal_mono_ns'] for r in sent), default=end)-end)/1e9),
        outstanding_at_cutoff=primary['pending_end'],
        dispatch_lag_ms=distribution([r['dispatch_lag_ns']/1e6 for r in rows if 'dispatch_lag_ns' in r]),
        send_lag_ms=distribution([(r['send_mono_ns']-origin-r['nominal_offset_ns'])/1e6 for r in sent]),
        wall_monotonic_offset_delta_ms=distribution([((r['send_ns']-capture['origin']['wall_ns'])-(r['send_mono_ns']-origin))/1e6 for r in sent]),
        scope='Fixed [30,150) time denominator, not start-to-last-response throughput. Sent-window cohort is followed through drain; failed/cancelled outcomes remain separate. Completion-window latency is never substituted. No automatic steady-state/capacity/SM-contention verdict; inspect all four blocks, due-unsent, pending, dispatch, rejection and CPU together.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resident', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if Path(args.cell).name != args.cell:
        parser.error('--cell is one directory name')
    source_hashes = {}

    def read(path):
        source_hashes[str(path)] = sha(path)
        return ([json.loads(line) for line in path.read_text().splitlines()]
                if path.suffix == '.jsonl' else json.loads(path.read_text()))

    cell = args.resident / 'cells' / args.cell
    manifest_path = cell / 'occurrence-manifest.json'
    manifest, capture = read(manifest_path), read(cell / 'sustained-capture.json')
    result = analyze_capture(capture, manifest)
    if capture['manifest_sha256'] != sha(manifest_path):
        result['issues'].append('Manifest byte SHA differs')
    run, ready = read(args.resident/'run.json'), read(args.resident/'cpu-sampler-ready.json')
    cpu_start = capture['origin']['wall_ns'] + SETTLING_SECONDS*10**9
    result['cpu'] = cpu_analysis(read(args.resident/'cpu-samples.jsonl'), ready, run,
                                cpu_start, cpu_start+(SEND_SECONDS-SETTLING_SECONDS)*10**9)
    result.update(source_sha256=source_hashes, analyzer_sha256=sha(__file__),
        quality_status='score actual occurrence WAVs separately', cell_id=args.cell)
    write_new(args.output, result)
    print(json.dumps({'cell': args.cell, 'issues': result['issues'], 'window': result['window']}))


if __name__ == '__main__':
    main()
