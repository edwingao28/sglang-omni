"""Round-to-round replication of the lane A contention sweep: sweep1/sweep1b (round 1) vs sweep2/sweep2b/sweep2c (round 2).

  evid_sweep_compare.py <results-root> <out-dir>     (reads <results-root>/summary/sustained-cells.json)
Writes sweep-replication.md: per arm and offered rate, completed succ/s in round 1 and round 2 (runaway-free,
collected cells only; a failure-stopped cell is kept and marked *), the difference, and per-arm max |diff|.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROUND = {'sweep1': 1, 'sweep1b': 1, 'sweep2': 2, 'sweep2b': 2, 'sweep2c': 2}
ARMS = ['DP1-graph-off', 'DP2-graph-off', 'DP2-graph-on', 'DP3-graph-off', 'DP3-graph-on', 'DP4-graph-off', 'DP4-graph-on']


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = json.loads((root / 'summary' / 'sustained-cells.json').read_text())['rows']
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['arm'] in ARMS and r['attempt'] in ROUND and r['placement'] is None and r['status'].startswith('collected') and r['runaway_successes'] == 0:
            cells[(r['arm'], r['rate'])][ROUND[r['attempt']]].append((r['success_per_s'], r['attempt'], 'failure_stop' in r['status']))
    lines = ['# Contention sweep replication (lane A, GPU 4): round 1 (`sweep1`/`sweep1b`) vs round 2 (`sweep2`/`sweep2b`/`sweep2c`)', '',
             'Completed succ/s in the [30,150) s window; runaway-free collected cells only; `*` = failure-stopped cell (kept, see CHECKPOINTS). '
             'Round 2 ran ~10 h after round 1 with the arm order rotated (DP1 → DP2+MPS → DP3+MPS → DP4+MPS → DP2 → DP3 → DP4 time-sliced).', '',
             '| arm | rate | round 1 | round 2 | diff (r2 − r1) |', '|---|---|---|---|---|']
    worst = {}
    for arm in ARMS:
        for rate in (8, 16, 24, 32, 40, 48):
            k = (arm, rate)
            if k not in cells:
                continue
            r1 = cells[k].get(1, []); r2 = cells[k].get(2, [])
            f = lambda xs: ' / '.join(f"{v:.2f}{'*' if fs else ''}" for v, a, fs in sorted(xs, key=lambda x: x[1]))
            diff = ''
            if r1 and r2:
                d = sum(v for v, _, _ in r2) / len(r2) - sum(v for v, _, _ in r1) / len(r1)
                diff = f'{d:+.2f}'
                worst[arm] = max(worst.get(arm, 0), abs(d))
            lines.append(f'| {arm} | {rate} | {f(r1)} | {f(r2)} | {diff} |')
    lines += ['', '## Max |round 2 − round 1| per arm (succ/s)', ''] + [f'- {arm}: {w:.2f}' for arm, w in worst.items()]
    (out / 'sweep-replication.md').write_text('\n'.join(lines) + '\n')
    print({arm: round(w, 2) for arm, w in worst.items()}, '->', out / 'sweep-replication.md')


if __name__ == '__main__':
    main()
