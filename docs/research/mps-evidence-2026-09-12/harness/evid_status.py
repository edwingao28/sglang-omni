"""Compact campaign status for the login node: python3 evid_status.py [results-root]

Per sustained run: status, cells, runaway successes (audio_duration_s >= 160) and occurrences of the
known runaway prompt; last lane-log lines; queue journal tails; trace runs.
"""
import glob
import json
import os
import sys
import time

_positional = [a for a in sys.argv[1:] if not a.startswith('--')]
R = _positional[0] if _positional else '/mnt/nfs/sa-shared/wenyao-minimax-h3/work/results/mps-evidence-20260912-a01/job-18012'
RUNAWAY_TEXT = 'Television can make'
now = time.time()


def cell_stats(cell_dir):
    try:
        cap = json.load(open(os.path.join(cell_dir, 'sustained-capture.json')))
    except Exception:
        return ''
    succ = [r for r in cap['per_request'] if r.get('outcome') == 'success']
    runaway = sum(1 for r in succ if float((r.get('result') or {}).get('audio_duration_s') or 0) >= 160)
    tv = sum(1 for r in succ if ((r.get('result') or {}).get('text') or '').startswith(RUNAWAY_TEXT))
    return f'succ={len(succ)} runaway={runaway}/{tv}'


if '--json' in sys.argv:
    snap = {}
    for run_json in glob.glob(f'{R}/sustained/*/run.json'):
        run = json.load(open(run_json))
        snap['run:' + run['run_id']] = run.get('status') + ' | ' + ' '.join(f"{c['cell_id'].rsplit('-', 1)[1]}:{c['status']}" for c in run.get('cells', [])) + (' | error: ' + str(run['error'])[:160] if run.get('error') else '')
    for run_json in glob.glob(f'{R}/runs/*/run.json'):
        run = json.load(open(run_json))
        snap['trace:' + run['run_id']] = str(run.get('status')) + (' post' if os.path.exists(os.path.join(R, 'analysis', run['run_id'], 'post-status.json')) else '')
    for q in glob.glob(f'{R}/queues/*.jsonl'):
        with open(q) as f:
            lines = f.readlines()
        if lines:
            last = json.loads(lines[-1])
            snap['queue:' + os.path.basename(q)] = f"line {last.get('line')} exit {last.get('exit')} {last.get('command', '')[:80]}"
    for lane in 'ABC':
        logs = sorted(glob.glob(f'{R}/lane-logs/{lane}-*.log'), key=os.path.getmtime)
        if logs:
            with open(logs[-1]) as f:
                tail = [l.rstrip() for l in f.readlines() if l.startswith('== queue')]
            if tail:
                snap['lane:' + lane] = tail[-1][:140]
    print(json.dumps(snap, sort_keys=True))
    raise SystemExit(0)

print('== sustained runs (recent first)')
for run_json in sorted(glob.glob(f'{R}/sustained/*/run.json'), key=os.path.getmtime, reverse=True):
    run = json.load(open(run_json))
    age_min = (now - os.path.getmtime(run_json)) / 60
    final = run.get('status') in ('selected_cells_collected_review_and_quality_pending', 'failed_preserved')
    cells = ' '.join(f"{c['cell_id'].rsplit('-', 1)[1]}:{c['status'].replace('collected', 'ok').replace('_with_failure_stop', '+stop')}"
                     + ((' [' + cell_stats(os.path.join(os.path.dirname(run_json), 'cells', c['cell_id'])) + ']') if not final or '--all' in sys.argv else '')
                     for c in run.get('cells', []))
    if final and age_min > 90 and '--all' not in sys.argv:
        continue
    print(f"{run['run_id']:38s} {run.get('status'):45s} {age_min:5.0f} min ago | {cells}")
    if run.get('error'):
        print('    error:', str(run['error'])[:200])
print('== traces (recent first)')
for run_json in sorted(glob.glob(f'{R}/runs/*/run.json'), key=os.path.getmtime, reverse=True)[:6]:
    run = json.load(open(run_json))
    post = os.path.join(R, 'analysis', run['run_id'], 'post-status.json')
    print(f"{run['run_id']:38s} {str(run.get('status')):32s} {(now - os.path.getmtime(run_json)) / 60:5.0f} min ago | post={'yes' if os.path.exists(post) else 'no'}")
print('== lane logs')
for lane in 'ABC':
    logs = sorted(glob.glob(f'{R}/lane-logs/{lane}-*.log'), key=os.path.getmtime)
    if logs:
        with open(logs[-1]) as f:
            tail = [l.rstrip() for l in f.readlines()[-2:]]
        print(f'{lane} {os.path.basename(logs[-1])}: ' + ' || '.join(t[:150] for t in tail))
print('== queue journals')
for q in sorted(glob.glob(f'{R}/queues/*.jsonl')):
    with open(q) as f:
        lines = f.readlines()
    if lines:
        last = json.loads(lines[-1])
        print(f"{os.path.basename(q):14s} line {last.get('line')} exit {last.get('exit')} {last.get('end')} | {last.get('command', '')[:90]}")
