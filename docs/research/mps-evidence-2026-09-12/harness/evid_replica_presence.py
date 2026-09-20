"""Per-replica kernel presence inside the measured client window (run on the login node, NFS):

  python3 evid_replica_presence.py <run_id> [<run_id> ...]

For each trace: device union presence, per-process (replica) union presence, Σ(per-replica)/device-union,
and the fraction of the window with >= 2 processes' kernels resident. Uses the client window from
analysis/<run>/trace-analysis.json when it carries one, else the kernel span. Writes
analysis/<run>/replica-presence.json.
"""
import json, sqlite3, sys
from pathlib import Path

R = Path('/mnt/nfs/sa-shared/wenyao-minimax-h3/work/results/mps-evidence-20260912-a01/job-18012')


def union_len(intervals, lo, hi):
    out, cs, ce = 0, None, None
    for s, e in intervals:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if ce is None or s > ce:
            if ce is not None:
                out += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        out += ce - cs
    return out


def window(run_id, rows):
    p = R / 'analysis' / run_id / 'trace-analysis.json'
    if p.exists():
        ta = json.loads(p.read_text())
        cw = ta.get('cuda_coverage_checks', {}).get('client_window_ns')
        if cw:
            return cw[0], cw[1], 'client_window'
        for key in ('client_window', 'window', 'measured_window'):
            w = ta.get(key) or ta.get('occupancy', {}).get(key)
            if isinstance(w, dict) and 'start_ns' in w and 'end_ns' in w:
                return w['start_ns'], w['end_ns'], key
        for key in ('client_window_start_ns', 'window_start_ns'):
            if key in ta:
                return ta[key], ta[key.replace('start', 'end')], key
    return rows[0][0], rows[-1][1], 'kernel-span'


def analyze(run_id):
    db = sqlite3.connect(f'file:{R}/profiles/{run_id}/capture.sqlite?mode=ro', uri=True)
    rows = db.execute('SELECT start, end, globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
    db.close()
    lo, hi, wkey = window(run_id, rows)
    w = hi - lo
    per = {}
    for s, e, pid in rows:
        per.setdefault(pid, []).append((s, e))
    per_presence = {str(pid): union_len(iv, lo, hi) / w for pid, iv in sorted(per.items())}
    device = union_len([(s, e) for s, e, _ in rows], lo, hi) / w
    # >=2 processes resident: sweep over per-process union intervals
    events = []
    for pid, iv in per.items():
        merged = []
        for s, e in iv:
            s, e = max(s, lo), min(e, hi)
            if e <= s:
                continue
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            events.append((s, 1)); events.append((e, -1))
    events.sort()
    depth, prev, multi = 0, None, 0
    for t, d in events:
        if depth >= 2 and prev is not None:
            multi += t - prev
        depth += d; prev = t
    out = dict(run_id=run_id, window_key=wkey, window_s=w / 1e9, processes=len(per), device_presence=device,
               per_process_presence=per_presence, sum_over_device=sum(per_presence.values()) / max(device, 1e-9),
               multi_process_resident=multi / w)
    (R / 'analysis' / run_id / 'replica-presence.json').write_text(json.dumps(out, indent=2) + '\n')
    return out


def main():
    print(f"{'trace':36s} win_s procs device%  per-process%            sum/device  >=2 resident%")
    for run_id in sys.argv[1:]:
        a = analyze(run_id)
        pp = '/'.join(f"{100 * v:.1f}" for v in a['per_process_presence'].values())
        print(f"{a['run_id']:36s} {a['window_s']:5.1f} {a['processes']:5d} {100 * a['device_presence']:6.1f}  {pp:22s} {a['sum_over_device']:9.2f}  {100 * a['multi_process_resident']:6.1f}   ({a['window_key']})")


if __name__ == '__main__':
    main()
