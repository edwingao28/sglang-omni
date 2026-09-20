"""Per-stage completion rates, queues and thread CPU from the pipeline event logs saved with each trace run.

  evid_stage_rates.py <results-root> [run_id ...]
Window = first request_admission .. last terminal_response. "mid" = central half of the window.
Preprocessing: completions/s over the whole window, over the mid-window, and over the stage's busy period
(first dispatch .. last stage_complete, i.e. the stage never ran dry in between when backlog stays > 0);
time-averaged counts: waiting for a worker (stage_dispatch -> host_handler_submit), in a worker
(host_handler_submit -> host_preprocess_service_exit); worker CPU per request (thread_cpu_ns delta over the
service interval); per-thread CPU utilisation over the window (thread_cpu delta / wall) for the replica's
pipeline process, summed = how much of one core the GIL-holding process burned.
Engine: time-averaged queue depth (scheduler_queue_enter -> model_path_start), running (model_path_start -> end).
Also writes 1 s bins (arrivals, preprocessing completions, preprocessing backlog, engine queue) per run.
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path


def load_events(run_dir):
    ev = []
    for p in sorted((run_dir / 'profiles' / 'events').glob('replica-*/events_*.jsonl')):
        with p.open() as fh:
            for line in fh:
                if line.strip():
                    e = json.loads(line)
                    e['replica'] = p.parent.name
                    ev.append(e)
    return ev


def by_req(ev, stage, name):
    return {e['request_id']: e for e in ev if e.get('stage') == stage and e['event_name'] == name}


def intervals(ev, stage, a, b):
    sa, sb = by_req(ev, stage, a), by_req(ev, stage, b)
    return [(sa[r]['timestamp_ns'], sb[r]['timestamp_ns'], sa[r], sb[r]) for r in sa if r in sb and sb[r]['timestamp_ns'] >= sa[r]['timestamp_ns']]


def mean_count(ivals, lo, hi):
    return sum(max(0, min(b, hi) - max(a, lo)) for a, b, *_ in ivals) / (hi - lo) if hi > lo else float('nan')


def count_at(ivals, t):
    return sum(a <= t < b for a, b, *_ in ivals)


def rate(ts, lo, hi):
    return sum(lo <= t < hi for t in ts) / ((hi - lo) / 1e9) if hi > lo else float('nan')


def p50(v):
    return statistics.median(v) if v else float('nan')


def pq(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float('nan')


def thread_cpu_util(ev, pids, lo, hi):
    """Per (pid, thread) CPU-seconds / wall over the window, from the thread_cpu_ns counters carried by events."""
    first, last = {}, {}
    for e in ev:
        if e.get('pid') in pids and 'thread_cpu_ns' in e and lo <= e['timestamp_ns'] <= hi:
            k = (e['pid'], e['thread_id'])
            if k not in first or e['timestamp_ns'] < first[k][0]:
                first[k] = (e['timestamp_ns'], e['thread_cpu_ns'])
            if k not in last or e['timestamp_ns'] > last[k][0]:
                last[k] = (e['timestamp_ns'], e['thread_cpu_ns'])
    out = {}
    for k in first:
        wall = last[k][0] - first[k][0]
        if wall > 0.5 * (hi - lo):  # only threads alive across most of the window
            out[k] = (last[k][1] - first[k][1]) / (hi - lo)
    return out


def analyze(run_dir):
    ev = load_events(run_dir)
    adm = sorted(e['timestamp_ns'] for e in ev if e['event_name'] == 'request_admission')
    term = [e['timestamp_ns'] for e in ev if e['event_name'] == 'terminal_response']
    if not adm or not term:
        return None
    lo, hi = adm[0], max(term)
    mlo, mhi = lo + (hi - lo) // 4, lo + 3 * (hi - lo) // 4
    replicas = sorted({e['replica'] for e in ev if e.get('stage') == 'preprocessing'})
    pre_stage = intervals(ev, 'preprocessing', 'stage_dispatch', 'stage_complete')
    pre_wait = intervals(ev, 'preprocessing', 'stage_dispatch', 'host_handler_submit')
    pre_work = intervals(ev, 'preprocessing', 'host_handler_submit', 'host_preprocess_service_exit')
    pre_serv = intervals(ev, 'preprocessing', 'host_preprocess_service_enter', 'host_preprocess_service_exit')
    ref_wait = intervals(ev, 'preprocessing', 'host_reference_future_wait_enter', 'host_reference_future_wait_exit')
    ref_batch = intervals(ev, 'preprocessing', 'host_reference_batch_enter', 'host_reference_batch_exit')
    q = intervals(ev, 'tts_engine', 'scheduler_queue_enter', 'model_path_start')
    run = intervals(ev, 'tts_engine', 'model_path_start', 'model_path_end')
    pre_done = sorted(b for _, b, *_ in pre_stage)
    busy_lo, busy_hi = min(a for a, *_ in pre_stage), max(pre_done)
    # 1 s bins
    bins = []
    t = lo
    while t < hi:
        bins.append(dict(t=(t - lo) / 1e9, arrivals=sum(t <= x < t + 10**9 for x in adm), pre_done=sum(t <= x < t + 10**9 for x in pre_done),
                         pre_backlog=count_at(pre_stage, t + 10**9), pre_in_worker=count_at(pre_work, t + 10**9),
                         eng_queue=count_at(q, t + 10**9), eng_running=count_at(run, t + 10**9)))
        t += 10**9
    # plateau check: completion rate in the 1 s bins where the stage had a backlog >= 8 waiting at the bin start
    loaded = [b['pre_done'] for b, prev in zip(bins[1:], bins[:-1]) if prev['pre_backlog'] - prev['pre_in_worker'] >= 8]
    pipeline_pids = {e['pid'] for e in ev if e.get('stage') == 'preprocessing'}
    cpu = thread_cpu_util(ev, pipeline_pids, mlo, mhi)
    worker_threads = {(x[3]['pid'], x[3]['thread_id']) for x in pre_serv}
    sched_threads = {(x[2]['pid'], x[2]['thread_id']) for x in run}
    ref_threads = {(x[2]['pid'], x[2]['thread_id']) for x in ref_batch}
    main_threads = {(e['pid'], e['thread_id']) for e in ev if e.get('stage') == 'preprocessing' and e['event_name'] == 'stage_dispatch'}
    def util(keys):
        return sum(v for k, v in cpu.items() if k in keys)
    out = dict(run_id=run_dir.name, replicas=len(replicas), window_s=(hi - lo) / 1e9, requests=len(adm),
               arrivals_per_s=len(adm) / ((hi - lo) / 1e9), arrivals_mid_per_s=rate(adm, mlo, mhi), arrival_span_s=(adm[-1] - adm[0]) / 1e9,
               pre_done_per_s=rate(pre_done, lo, hi), pre_done_mid_per_s=rate(pre_done, mlo, mhi),
               pre_busy_s=(busy_hi - busy_lo) / 1e9, pre_done_busy_per_s=len(pre_done) / ((busy_hi - busy_lo) / 1e9),
               pre_loaded_bins=len(loaded), pre_done_loaded_per_s=(sum(loaded) / len(loaded) if loaded else float('nan')),
               pre_backlog_max=max(b['pre_backlog'] for b in bins),
               pre_wait_mid=mean_count(pre_wait, mlo, mhi), pre_in_worker_mid=mean_count(pre_work, mlo, mhi), pre_in_worker_max=max(b['pre_in_worker'] for b in bins),
               pre_stage_p50_ms=p50([(b - a) / 1e6 for a, b, *_ in pre_stage]), pre_wait_p50_ms=p50([(b - a) / 1e6 for a, b, *_ in pre_wait]),
               pre_service_p50_ms=p50([(b - a) / 1e6 for a, b, *_ in pre_serv]), pre_service_p95_ms=pq([(b - a) / 1e6 for a, b, *_ in pre_serv], 0.95),
               ref_wait_p50_ms=p50([(b - a) / 1e6 for a, b, *_ in ref_wait]), ref_batch_p50_ms=p50([(b - a) / 1e6 for a, b, *_ in ref_batch]),
               ref_batches=len(ref_batch), ref_batch_cpu_p50_ms=p50([(y['thread_cpu_ns'] - x['thread_cpu_ns']) / 1e6 for _, _, x, y in ref_batch]),
               pre_worker_cpu_p50_ms=p50([(y['thread_cpu_ns'] - x['thread_cpu_ns']) / 1e6 for _, _, x, y in pre_serv]),
               cpu_workers=util(worker_threads), cpu_ref=util(ref_threads), cpu_sched=util(sched_threads), cpu_main=util(main_threads), cpu_all=sum(cpu.values()), cpu_threads=len(cpu),
               engine_queue_mid=mean_count(q, mlo, mhi), engine_running_mid=mean_count(run, mlo, mhi),
               queue_p50_ms=p50([(b - a) / 1e6 for a, b, *_ in q]), model_p50_ms=p50([(b - a) / 1e6 for a, b, *_ in run]), bins=bins)
    return out


COLS = [('run', 'run_id', '{}', 44), ('rep', 'replicas', '{}', 3), ('win', 'window_s', '{:.1f}', 5), ('arr', 'arrivals_per_s', '{:.1f}', 5), ('arr mid', 'arrivals_mid_per_s', '{:.1f}', 7),
        ('done', 'pre_done_per_s', '{:.1f}', 5), ('done mid', 'pre_done_mid_per_s', '{:.1f}', 8), ('busy s', 'pre_busy_s', '{:.1f}', 6), ('done busy', 'pre_done_busy_per_s', '{:.1f}', 9),
        ('ld bins', 'pre_loaded_bins', '{}', 7), ('done ld', 'pre_done_loaded_per_s', '{:.1f}', 7), ('bl max', 'pre_backlog_max', '{}', 6), ('wait', 'pre_wait_mid', '{:.1f}', 5), ('wrk', 'pre_in_worker_mid', '{:.1f}', 4), ('wrk mx', 'pre_in_worker_max', '{}', 6),
        ('wait p50', 'pre_wait_p50_ms', '{:.0f}', 8), ('svc p50', 'pre_service_p50_ms', '{:.0f}', 7), ('svc p95', 'pre_service_p95_ms', '{:.0f}', 7), ('refw p50', 'ref_wait_p50_ms', '{:.0f}', 8), ('refb p50', 'ref_batch_p50_ms', '{:.0f}', 8), ('refb n', 'ref_batches', '{}', 6),
        ('wcpu ms', 'pre_worker_cpu_p50_ms', '{:.0f}', 7), ('rcpu ms', 'ref_batch_cpu_p50_ms', '{:.0f}', 7),
        ('cpu wrk', 'cpu_workers', '{:.2f}', 7), ('cpu ref', 'cpu_ref', '{:.2f}', 7), ('cpu sch', 'cpu_sched', '{:.2f}', 7), ('cpu main', 'cpu_main', '{:.2f}', 8), ('cpu all', 'cpu_all', '{:.2f}', 7), ('thr', 'cpu_threads', '{}', 3),
        ('eq', 'engine_queue_mid', '{:.1f}', 5), ('er', 'engine_running_mid', '{:.1f}', 5), ('q p50', 'queue_p50_ms', '{:.0f}', 5), ('m p50', 'model_p50_ms', '{:.0f}', 5)]


def main():
    root = Path(sys.argv[1])
    ids = sys.argv[2:] or sorted(p.name for p in (root / 'runs').glob('*-trace-*'))
    rows = [r for r in (analyze(root / 'runs' / rid) for rid in ids if (root / 'runs' / rid / 'profiles' / 'events').exists()) if r]
    print(' | '.join(f'{h:>{w}s}' if i else f'{h:{w}s}' for i, (h, _, _, w) in enumerate(COLS)))
    for r in rows:
        print(' | '.join((f.format(r[k]) if i else r[k].replace('-n128', '')).rjust(w) if i else r[k].replace('-n128', '').ljust(w) for i, (h, k, f, w) in enumerate(COLS)))
    out = root / 'summary'
    out.mkdir(exist_ok=True)
    (out / 'stage-rates.json').write_text(json.dumps(rows, indent=1) + '\n')
    print('->', out / 'stage-rates.json')


if __name__ == '__main__':
    main()
