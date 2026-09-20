"""Per-kernel duration statistics from trace sqlite files (run on the login node, NFS):

  python3 evid_kernel_durations.py <run_id> [<run_id> ...]   (first run = baseline for ratios)

Purpose: the trace-level "cross-replica overlap" cannot separate true concurrent execution (MPS)
from mid-kernel time-slice preemption (two contexts without MPS), because CUPTI reports one
start..end interval per kernel either way. Preemption inflates the duration of the preempted
kernel by the other context's slice; concurrency on an under-occupied GPU should not. So compare,
per kernel name, the durations of kernels that overlap another process's kernel with those that
do not, and both against the single-replica baseline.
"""
import json
import sqlite3
import statistics
import sys

R = '/mnt/nfs/sa-shared/wenyao-minimax-h3/work/results/mps-evidence-20260912-a01/job-18012'


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float('nan')


def load(run_id):
    db = sqlite3.connect(f'file:{R}/profiles/{run_id}/capture.sqlite?mode=ro', uri=True)
    rows = db.execute('SELECT k.start, k.end, k.globalPid, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                      'JOIN StringIds s ON s.id = k.shortName ORDER BY k.start').fetchall()
    db.close()
    return rows


def analyze(run_id):
    rows = load(run_id)
    pids = sorted({r[2] for r in rows})
    # sweep: does each kernel overlap a kernel of another pid?
    active = []  # (end, pid) of kernels still running
    overlapped = [False] * len(rows)
    import heapq
    heap = []
    for i, (start, end, pid, name) in enumerate(rows):
        while heap and heap[0][0] <= start:
            heapq.heappop(heap)
        for other_end, other_pid, j in heap:
            if other_pid != pid and other_end > start:
                overlapped[i] = True
                overlapped[j] = True
        heapq.heappush(heap, (end, pid, i))
    # union of kernel intervals
    union = 0
    cur_s, cur_e = None, None
    for start, end, _, _ in rows:
        if cur_e is None or start > cur_e:
            if cur_e is not None:
                union += cur_e - cur_s
            cur_s, cur_e = start, end
        else:
            cur_e = max(cur_e, end)
    if cur_e is not None:
        union += cur_e - cur_s
    total = sum(end - start for start, end, _, _ in rows)
    by_name = {}
    for i, (start, end, pid, name) in enumerate(rows):
        entry = by_name.setdefault(name, {'all': [], 'ov': [], 'solo': []})
        dur = end - start
        entry['all'].append(dur)
        entry['ov' if overlapped[i] else 'solo'].append(dur)
    return dict(run_id=run_id, kernels=len(rows), pids=len(pids), span_s=(rows[-1][1] - rows[0][0]) / 1e9,
                total_kernel_s=total / 1e9, union_s=union / 1e9, sum_over_union=total / max(union, 1),
                overlapped_fraction=sum(overlapped) / max(len(rows), 1), by_name=by_name)


def main():
    runs = [analyze(r) for r in sys.argv[1:]]
    base = runs[0]
    print(f"{'trace':34s} kernels pids span_s  sum_s  union_s sum/union overlapped_kernels")
    for a in runs:
        print(f"{a['run_id']:34s} {a['kernels']:7d} {a['pids']:4d} {a['span_s']:6.1f} {a['total_kernel_s']:6.2f} {a['union_s']:7.2f} {a['sum_over_union']:8.2f} {100 * a['overlapped_fraction']:6.1f}%")
    top = sorted(base['by_name'], key=lambda n: -sum(base['by_name'][n]['all']))[:14]
    print('\nper kernel name (top by total time in the baseline): p50 duration in µs; ov = overlapping another process, solo = not')
    head = f"{'kernel':40s} " + ' | '.join(f"{a['run_id'].replace('-trace', '').replace('-n128', ''):>34s}" for a in runs)
    print(head)
    for name in top:
        cells = []
        for a in runs:
            e = a['by_name'].get(name)
            if not e:
                cells.append(f"{'-':>34s}")
                continue
            cells.append(f"n={len(e['all']):6d} p50={pct(e['all'], .5) / 1e3:7.1f} ov={pct(e['ov'], .5) / 1e3 if e['ov'] else float('nan'):7.1f} solo={pct(e['solo'], .5) / 1e3 if e['solo'] else float('nan'):6.1f}")
        print(f"{name[:40]:40s} " + ' | '.join(cells))
    out = {a['run_id']: {k: v for k, v in a.items() if k != 'by_name'} | {'top_kernels': {n: {k: dict(n=len(v), p50_us=pct(v, .5) / 1e3, p90_us=pct(v, .9) / 1e3) for k, v in a['by_name'][n].items() if v} for n in top if n in a['by_name']}} for a in runs}
    path = f"{R}/analysis/kernel-durations-{'_vs_'.join(r.replace('-trace-t1-n128', '') for r in sys.argv[1:])}.json"
    json.dump(out, open(path, 'w'), indent=1)
    print('\nwritten', path)


if __name__ == '__main__':
    main()
