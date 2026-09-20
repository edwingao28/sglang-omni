"""Summarize sustained cells (window flow, latency, telemetry, CPU roles) into JSON + Markdown.

  evid_summarize.py <results-root> <out-dir>
<results-root> mirrors RESULT_ROOT (needs sustained/<run>/{run.json,cells/*,cpu-samples.jsonl,
cpu-sampler-ready.json,gpu-telemetry-*.log}). Descriptive only: no capacity verdicts are
computed; the table exposes the quantities a reader needs to judge saturation (pending
growth, rejections, block-to-block completion rates) and the host side (CPU per role).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_load_cell import cpu_analysis  # noqa: E402
from sustained_analyzer import analyze_capture  # noqa: E402

SETTLING_S, SEND_S = 30, 150
RUNAWAY_AUDIO_S = 160.0  # talker cap 2048 frames = 163.84 s; normal outputs are 3-20 s


def parse_telemetry(path, start_ns, end_ns):
    rows = []
    with path.open() as stream:
        reader = csv.reader(stream)
        header = [h.strip() for h in next(reader)]
        for row in reader:
            if len(row) != len(header):
                continue
            rec = dict(zip(header, [c.strip() for c in row]))
            try:
                ts = datetime.strptime(rec['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            ns = int(ts.timestamp() * 1e9)
            if start_ns <= ns < end_ns:
                rows.append(rec)
    def num(key):
        out = []
        for rec in rows:
            value = rec.get(key, '').split()[0] if rec.get(key) else ''
            try:
                out.append(float(value))
            except ValueError:
                pass
        return out
    clocks, power, util, mem, memutil = (num(k) for k in ('clocks.current.sm [MHz]', 'power.draw [W]', 'utilization.gpu [%]', 'memory.used [MiB]', 'utilization.memory [%]'))
    return dict(samples=len(rows), sm_clock_mhz_mean=statistics.fmean(clocks) if clocks else None,
                sm_clock_below_1900_fraction=sum(c < 1900 for c in clocks) / len(clocks) if clocks else None,
                power_w_mean=statistics.fmean(power) if power else None,
                util_gpu_pct_mean=statistics.fmean(util) if util else None,
                util_mem_pct_mean=statistics.fmean(memutil) if memutil else None,
                memory_used_mib_max=max(mem) if mem else None)


def summarize_cell(run_dir, run, cell):
    cell_dir = run_dir / 'cells' / cell['cell_id']
    capture = json.loads((cell_dir / 'sustained-capture.json').read_text())
    manifest = json.loads((cell_dir / 'occurrence-manifest.json').read_text())
    flow = analyze_capture(capture, manifest)  # recomputed with the current analyzer; run-time copy stays in the cell dir
    recorded = json.loads((cell_dir / 'flow-analysis.json').read_text())
    flow['issues_recorded_at_run_time'] = recorded.get('issues', [])
    origin = capture['origin']['wall_ns']
    start, end = origin + SETTLING_S * 10**9, origin + SEND_S * 10**9
    row = dict(run_id=run['run_id'], arm=run['arm'], attempt=run['attempt'], placement=run.get('placement'), sms=run.get('sms'),
               replicas=run['replicas'], mps=run['mps'], graph=run['graph'], cores_per_replica=run['cores_per_replica'],
               cell_id=cell['cell_id'], rate=cell['rate'], status=cell['status'], flow_issues=flow['issues'],
               flow_issues_at_run_time=flow['issues_recorded_at_run_time'], late_sends_within_grace=flow.get('sends_after_nominal_cutoff_within_grace'),
               lane_gpu=run['gpu_uuid'], window_start_utc=datetime.fromtimestamp(start / 1e9, timezone.utc).isoformat())
    w = flow['window']
    row.update(offered_per_s=w['actual_offered_rate'], success_per_s=w['successful_completion_rate'],
               terminal_per_s=w['all_terminal_rate'], window_sends=w['actual_sends'], window_success=w['success_completions'],
               window_rejections=w['terminal_outcomes'].get('admission_rejection', 0),
               window_other_terminals={k: v for k, v in w['terminal_outcomes'].items() if k not in ('success', 'admission_rejection')},
               pending_start=w['pending_start'], pending_end=w['pending_end'], pending_change=w['pending_change'],
               due_unsent_end=w['due_unsent_end'], outstanding_peak=w['client_transaction_outstanding']['peak'],
               block_success_per_s=[b['successful_completion_rate'] for b in flow['blocks']],
               block_rejections=[b['terminal_outcomes'].get('admission_rejection', 0) for b in flow['blocks']],
               drain_s=flow['drain_s'], dispatch_lag_p95_ms=flow['dispatch_lag_ms']['p95'] if flow['dispatch_lag_ms'] else None)
    # Runaway generations: successes whose audio reached the talker cap (2048 codec frames x 80 ms = 163.84 s).
    succ_results = [r.get('result') or {} for r in capture['per_request'] if r.get('outcome') == 'success']
    durations = sorted(float(x['audio_duration_s']) for x in succ_results if x.get('audio_duration_s') not in (None, ''))
    row.update(runaway_successes=sum(d >= RUNAWAY_AUDIO_S for d in durations),
               audio_s_p50=durations[len(durations) // 2] if durations else None, audio_s_max=durations[-1] if durations else None,
               success_results_with_duration=len(durations))
    cohort = flow['sent_window_cohort']
    lat = cohort.get('success_latency_s') or {}
    row.update(cohort_sent=cohort['count'], cohort_success=cohort['success_count'],
               cohort_outcomes=cohort['outcomes'], latency_p50_s=lat.get('p50'), latency_p95_s=lat.get('p95'),
               latency_mean_s=lat.get('mean'), latency_max_s=lat.get('max'))
    tele = sorted(run_dir.glob('gpu-telemetry-*.log'))
    row['telemetry'] = parse_telemetry(tele[0], start, end) if tele else None
    quality = cell_dir / 'sustained-quality' / 'quality-audit.json'
    if quality.exists():
        qa = json.loads(quality.read_text())
        row['quality'] = dict(corpus_wer=qa.get('corpus_wer'), scored=qa.get('scored_occurrences'), unscored=qa.get('unscored_occurrences'),
                              reference_words=qa.get('reference_words'), errors=qa.get('errors'))
    else:
        row['quality'] = None
    try:
        samples = [json.loads(line) for line in (run_dir / 'cpu-samples.jsonl').read_text().splitlines() if line.strip()]
        ready = json.loads((run_dir / 'cpu-sampler-ready.json').read_text())
        cpu = cpu_analysis(samples, ready, run, start, end)
        row['cpu'] = dict(covered_window_fraction=cpu['covered_window_fraction'], issues=cpu['issues'],
                          mps_ownership_verified=cpu['mps_ownership_verified'],
                          roles={r['role']: dict(one_core_pct=r['mean_one_core_pct_over_covered_window'],
                                                 declared_capacity_pct=r['mean_declared_capacity_pct'],
                                                 declared_cpus=len(r['declared_cpu_set']) if r['declared_cpu_set'] else None)
                                 for r in cpu['roles']},
                          intervals_with_missing_required=cpu['process_intervals_with_missing_required'])
    except Exception as error:  # noqa: BLE001 - keep the flow numbers even if CPU samples are unusable
        row['cpu'] = dict(error=repr(error))
    return row


def fmt(value, digits=1):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def markdown(rows):
    head = ['arm', 'attempt', 'rate', 'offered/s', 'succ/s', 'term/s', 'rej', 'pend Δ', 'runaway', 'blocks succ/s', 'p50 s', 'p95 s', 'drain s',
            'SM MHz', 'W', 'util%', 'replica CPU% (1-core)', 'router%', 'client%', 'mps%', 'WER% (scored)', 'issues']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        cpu = r.get('cpu') or {}
        roles = cpu.get('roles') or {}
        reps = [roles[k]['one_core_pct'] for k in sorted(roles) if k.startswith('replica-')]
        mps = sum(roles[k]['one_core_pct'] or 0 for k in roles if k.startswith('mps-'))
        tele = r.get('telemetry') or {}
        issues = list(r['flow_issues']) + (['cpu:' + i for i in cpu.get('issues', [])] if cpu.get('issues') else []) + (['cpu-error'] if 'error' in cpu else [])
        lines.append('| ' + ' | '.join([
            r['arm'] + (f"/{r['placement']}{r['sms'] or ''}" if r.get('placement') else ''), r['attempt'], fmt(r['rate'], 0), fmt(r['offered_per_s'], 2),
            fmt(r['success_per_s'], 2), fmt(r['terminal_per_s'], 2), str(r['window_rejections']), f"{r['pending_change']:+d}",
            str(r.get('runaway_successes', '')), '/'.join(fmt(b, 1) for b in r['block_success_per_s']), fmt(r['latency_p50_s'], 2), fmt(r['latency_p95_s'], 2), fmt(r['drain_s'], 1),
            fmt(tele.get('sm_clock_mhz_mean'), 0), fmt(tele.get('power_w_mean'), 0), fmt(tele.get('util_gpu_pct_mean'), 0),
            '/'.join(fmt(x, 0) for x in reps), fmt(roles.get('router', {}).get('one_core_pct'), 0), fmt(roles.get('client', {}).get('one_core_pct'), 0),
            fmt(mps, 0) if mps else '', (f"{100 * r['quality']['corpus_wer']:.2f} ({r['quality']['scored']})" if r.get('quality') and r['quality'].get('corpus_wer') is not None else ''),
            '; '.join(issues)]) + ' |')
    return '\n'.join(lines)


def pivot(rows):
    """Median success/s per (arm, placement, rate) across attempts, with attempt count and spread."""
    groups = {}
    for r in rows:
        key = (r['arm'] + (f"/{r['placement']}{r['sms'] or ''}" if r.get('placement') else ''), r['rate'])
        groups.setdefault(key, []).append(r)
    out = []
    for (arm, rate), items in sorted(groups.items()):
        succ = [i['success_per_s'] for i in items]
        p95 = [i['latency_p95_s'] for i in items if i['latency_p95_s'] is not None]
        out.append(dict(arm=arm, rate=rate, n=len(items), attempts=[i['attempt'] for i in items],
                        success_per_s_median=statistics.median(succ), success_per_s_min=min(succ), success_per_s_max=max(succ),
                        rejections_median=statistics.median(i['window_rejections'] for i in items),
                        runaway_total=sum(i.get('runaway_successes') or 0 for i in items), success_total=sum(i['window_success'] for i in items),
                        pending_change_median=statistics.median(i['pending_change'] for i in items),
                        latency_p95_s_median=statistics.median(p95) if p95 else None))
    return out


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for run_json in sorted(root.glob('sustained/*/run.json')):
        run = json.loads(run_json.read_text())
        for cell in run.get('cells', []):
            if cell.get('status') not in ('collected', 'collected_with_failure_stop'):
                skipped.append(dict(run_id=run['run_id'], cell=cell.get('cell_id'), status=cell.get('status')))
                continue
            try:
                rows.append(summarize_cell(run_json.parent, run, cell))
            except Exception as error:  # noqa: BLE001
                skipped.append(dict(run_id=run['run_id'], cell=cell.get('cell_id'), error=repr(error)))
    rows.sort(key=lambda r: (r['replicas'], r['arm'], r.get('placement') or '', r.get('sms') or 0, r['rate'], r['attempt']))
    (out / 'sustained-cells.json').write_text(json.dumps(dict(rows=rows, skipped=skipped, pivot=pivot(rows)), indent=2, default=str) + '\n')
    text = ['# Sustained cells (fixed [30,150) s window; open-loop Poisson, seed 42)', '', markdown(rows), '',
            '## Median success/s per arm × rate', '', '| arm | rate | n | succ/s median (min–max) | rej median | pend Δ median | p95 s median | runaway / successes |', '|---|---|---|---|---|---|---|---|']
    for p in pivot(rows):
        text.append(f"| {p['arm']} | {p['rate']:g} | {p['n']} | {p['success_per_s_median']:.2f} ({p['success_per_s_min']:.2f}–{p['success_per_s_max']:.2f}) | {p['rejections_median']:g} | {p['pending_change_median']:+g} | {fmt(p['latency_p95_s_median'], 2)} | {p['runaway_total']} / {p['success_total']} |")
    if skipped:
        text += ['', '## Skipped cells', ''] + [f"- {json.dumps(s)}" for s in skipped]
    (out / 'sustained-cells.md').write_text('\n'.join(text) + '\n')
    print(f'{len(rows)} cells summarized, {len(skipped)} skipped -> {out}')


if __name__ == '__main__':
    main()
