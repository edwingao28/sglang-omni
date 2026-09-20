"""Tabulate trace analyses (analysis/<run_id>/*.json) into JSON + Markdown.

  evid_trace_table.py <results-root> <out-dir>
Columns: kernel presence (union) and cross-replica overlap as fractions of the measured
client window, zero-kernel time split by cause (host launch not yet completed vs. launch
completed), request-level stage medians, and 1 kHz GPU metrics when captured.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


def pct(x):
    return '' if x is None else f'{100 * x:.1f}'


def med(values):
    values = sorted(values)
    return values[len(values) // 2] if values else None


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted((root / 'analysis').glob('*/')):
        ta_path = d / 'trace-analysis.json'
        if not ta_path.exists():
            continue
        ta = json.loads(ta_path.read_text())
        occ, w = ta['occupancy'], ta['occupancy']['window_ns']
        row = dict(run_id=ta['run_id'], validity=ta['validity'], requests=ta['requests'], window_s=w / 1e9,
                   kernel_present=occ['kernel_union_ns'] / w, cross_replica_overlap=(occ.get('cross_replica_overlap_ns') or 0) / w,
                   zero_kernel=occ['zero_kernel_ns'] / w, gap_p50_us=ta['gap_duration_ns'].get('median', 0) / 1e3,
                   gap_p99_us=ta['gap_duration_ns'].get('p99', 0) / 1e3, gap_max_ms=ta['gap_duration_ns'].get('max', 0) / 1e6,
                   diagnostic_warnings=len(ta['diagnostic_warnings']), device_only=ta['cuda_coverage_checks']['expected_device_only'],
                   graph_nodes_in_window=ta['graph_node_kernel_count_in_client_window'], kernels_in_window=ta['kernel_count_in_client_window'])
        si = ta['stage_intervals']
        row['model_path_ms_p50'] = med([x['wall_ns'] / 1e6 for x in si.get('tts_engine:model_path_start->model_path_end', [])])
        row['queue_wait_ms_p50'] = med([x['wall_ns'] / 1e6 for x in si.get('tts_engine:scheduler_queue_enter->model_path_start', [])])
        row['e2e_ms_p50'] = med([x['wall_ns'] / 1e6 for x in si.get('coordinator:request_admission->terminal_response', [])])
        results_csv = root / 'runs' / ta['run_id'] / 'metrics' / 'results.csv'
        if results_csv.exists():
            with results_csv.open() as stream:
                durations = [float(x['audio_duration_s']) for x in csv.DictReader(stream) if x.get('audio_duration_s')]
            row['runaway_requests'] = sum(dur >= 160 for dur in durations)  # talker cap 2048 frames = 163.84 s
        sg_path = d / 'submission-gaps.json'
        if sg_path.exists():
            sg = json.loads(sg_path.read_text())['submission_gap_analysis']
            frac = sg['fraction_of_zero_kernel_time_by_class']
            row['zero_host_launch_pending'] = frac.get('before_next_recorded_future_launch_completion', 0) * row['zero_kernel']
            row['zero_launch_completed'] = frac.get('recorded_future_kernel_launch_completed', 0) * row['zero_kernel']
        gm_path = d / 'gpu-metrics.json'
        if gm_path.exists():
            gm = json.loads(gm_path.read_text())
            if gm.get('status') == 'summarized':
                m = {k.split(' [')[0]: v for k, v in gm['metrics'].items()}
                for key, name in (('sms_active', 'SMs Active'), ('sm_issue', 'SM Issue'), ('tensor_active', 'Tensor Active'),
                                  ('gr_active', 'GR Active'), ('dram_read', 'DRAM Read Bandwidth')):
                    if name in m:
                        row[key] = dict(mean=m[name]['mean'], p50=m[name]['p50'], p95=m[name]['p95'], zero_fraction=m[name]['zero_fraction'])
        ts_path = d / 'gpu-metrics.timeseries.csv'
        if ts_path.exists():  # central half of the measured window: steady state without the arrival ramp and the drain
            with ts_path.open() as stream:
                ts_rows = list(csv.DictReader(stream))
            if ts_rows:
                t_end = float(ts_rows[-1]['t_offset_s'])
                mid = [r for r in ts_rows if 0.25 * t_end <= float(r['t_offset_s']) <= 0.75 * t_end]
                for key, name in (('sms_active', 'SMs Active'), ('sm_issue', 'SM Issue'), ('gr_active', 'GR Active')):
                    col = next((k for k in ts_rows[0] if k.startswith(name)), None)
                    if col and mid and row.get(key):
                        row[key]['mid50'] = sum(float(r[col]) for r in mid) / len(mid)
        gc_path = d / 'gc-masks.json'
        if gc_path.exists():
            gc = json.loads(gc_path.read_text())
            greens = [w.get('green_busy_fraction_of_owned') for w in gc.get('workers', []) if w.get('green_busy_fraction_of_owned') is not None]
            row['gc_masks'] = dict(status=gc['status'], issues=gc['issues'][:3], verdict={k: v for k, v in gc['tpc_mask_verdict'].items() if k != 'pairwise'},
                                   pairwise=gc['tpc_mask_verdict'].get('pairwise'), green_busy_fraction_of_owned=greens,
                                   shared_tpcs=[x['shared_tpcs'] for x in gc['tpc_mask_verdict'].get('pairwise') or []])
        rows.append(row)
    (out / 'trace-summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    head = ['trace', 'validity', 'window s', 'runaway req', 'kernel present %', 'cross-replica overlap %', 'zero: host launch pending %', 'zero: launch done %',
            'gap p50 µs', 'gap p99 µs', 'queue wait p50 ms', 'model path p50 ms', 'e2e p50 ms', 'SMs Active % (p50/p95)', 'SMs Active mid-window %', 'SM Issue %', 'Tensor %', 'GR Active %', 'GC masks']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        sa = r.get('sms_active')
        gc = r.get('gc_masks')
        lines.append('| ' + ' | '.join([
            r['run_id'].replace('-n128', ''), r['validity'].replace('_', ' '), f"{r['window_s']:.1f}", str(r.get('runaway_requests', '')), pct(r['kernel_present']), pct(r['cross_replica_overlap']),
            pct(r.get('zero_host_launch_pending')), pct(r.get('zero_launch_completed')), f"{r['gap_p50_us']:.1f}", f"{r['gap_p99_us']:.0f}",
            '' if r['queue_wait_ms_p50'] is None else f"{r['queue_wait_ms_p50']:.0f}", '' if r['model_path_ms_p50'] is None else f"{r['model_path_ms_p50']:.0f}",
            '' if r['e2e_ms_p50'] is None else f"{r['e2e_ms_p50']:.0f}",
            '' if not sa else f"{sa['mean']:.1f} ({sa['p50']:.0f}/{sa['p95']:.0f})",
            '' if not sa or 'mid50' not in sa else f"{sa['mid50']:.1f}",
            '' if not r.get('sm_issue') else f"{r['sm_issue']['mean']:.1f}", '' if not r.get('tensor_active') else f"{r['tensor_active']['mean']:.1f}",
            '' if not r.get('gr_active') else f"{r['gr_active']['mean']:.1f}",
            '' if not gc else (gc['status'] + (f" disjoint={gc['verdict'].get('all_disjoint')} shared={'/'.join(map(str, gc['shared_tpcs']))} union_tpcs={gc['verdict'].get('union_tpcs')} green_busy={'/'.join(f'{100 * g:.0f}' for g in gc['green_busy_fraction_of_owned'])}%" if gc['verdict'].get('replicas_with_green') else ''))]) + ' |')
    (out / 'trace-summary.md').write_text('# Trace summary (Nsight, 128-request measured windows)\n\n' + '\n'.join(lines) + '\n\n'
        'Kernel present = union of kernel intervals over the client window (presence, not occupancy). '
        'Zero-kernel causes from analyze_submission_gaps: "host launch pending" = the next recorded kernel\'s launch API had not returned yet; '
        '"launch done" = launch returned, GPU still idle. GPU metrics are device-wide 1 kHz samples (nsys gh100 set); "mid-window" = mean over the central half of the window ([0.25, 0.75] × window), without the arrival ramp and the drain.\n')
    print(f'{len(rows)} traces -> {out}')


if __name__ == '__main__':
    main()
