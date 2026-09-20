"""Green Context placement figure: succ/s and p95 latency per offered rate for each DP3 placement.

  evid_fig_gc.py <results-root> <out-dir>     (reads <results-root>/summary/sustained-cells.json)
Attempts gc1..gc3 are drawn as separate markers of the same colour; the unpartitioned sweep
(attempt sweep1, no adapter) is the reference.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {None: 'no adapter (sweep1)', 'ordinary': 'adapter, ordinary streams (132 SMs each)',
          'indexed40': 'indexed 3 × 40 SMs (disjoint)', 'indexed32': 'indexed 3 × 32 SMs (disjoint)',
          'union240': 'union2 3 × 80 SMs (overlapping)', 'union2-40': 'union2 3 × 80 SMs (overlapping)'}
MARKERS = {'sweep1': 'o', 'gc1': 's', 'gc2': '^', 'gc3': 'v'}


def key(row):
    if row['placement'] is None:
        return None
    return f"{row['placement']}{row['sms'] or ''}"


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = json.loads((root / 'summary' / 'sustained-cells.json').read_text())['rows']
    rows = [r for r in rows if r['arm'] == 'DP3-graph-on' and r['status'].startswith('collected')
            and (r['attempt'].startswith('gc') or (r['attempt'] == 'sweep1' and r['placement'] is None))]
    if not rows:
        print('no GC rows'); return
    groups = defaultdict(list)
    for r in rows:
        groups[(key(r), r['attempt'])].append(r)
    placements = sorted({k for k, _ in groups}, key=lambda k: ['', 'ordinary', 'indexed40', 'indexed32', 'union240'].index(k or ''))
    colors = {p: c for p, c in zip(placements, plt.rcParams['axes.prop_cycle'].by_key()['color'])}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for (p, attempt), cells in sorted(groups.items(), key=lambda kv: (placements.index(kv[0][0]), kv[0][1])):
        cells = sorted(cells, key=lambda r: r['rate'])
        x = [r['rate'] for r in cells]
        ax1.plot(x, [r['success_per_s'] for r in cells], marker=MARKERS.get(attempt, 'x'), color=colors[p], ls='-' if attempt in ('sweep1', 'gc1') else '--',
                 label=f"{LABELS.get(p, p)} [{attempt}]")
        ax2.plot(x, [r['latency_p95_s'] for r in cells], marker=MARKERS.get(attempt, 'x'), color=colors[p], ls='-' if attempt in ('sweep1', 'gc1') else '--')
    ax1.plot([0, 48], [0, 48], color='grey', lw=0.6, ls=':')
    ax1.set_xlabel('offered rate (req/s, Poisson)'); ax1.set_ylabel('completed succ/s in [30,150) s'); ax1.set_title('DP3 graph + MPS: throughput per placement')
    ax2.set_xlabel('offered rate (req/s, Poisson)'); ax2.set_ylabel('p95 latency (s)'); ax2.set_yscale('log'); ax2.set_title('p95 latency')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.set_xlim(0, 50)
    ax1.legend(fontsize=7, loc='lower right')
    fig.suptitle('Experiment 2.2 — Green Context partitions (one H100, 3 replicas, MPS on; alignment 8 → 40×3 / 32×3 / union2 80)', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'gc-placements.{ext}', dpi=150)
    print('placements:', [LABELS.get(p, p) for p in placements], '->', out / 'gc-placements.png')


if __name__ == '__main__':
    main()
