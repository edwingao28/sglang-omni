"""Green Context partition cost across replica counts: saturated succ/s per placement at DP2/DP3/DP4.

  evid_fig_gc_cost.py <results-root> <out-dir>     (reads <results-root>/summary/sustained-cells.json)
Bars = mean over rounds, markers = individual rounds; the label above each bar is the change vs the
unpartitioned MPS-on service at the same offered rate. Cells with a runaway success are excluded.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAT_RATE = {2: 32, 3: 40, 4: 48}
REF_ATTEMPTS = {2: ('sweep1', 'sweepB'), 3: ('sweep1',), 4: ('sweep1b',)}
COLORS = {None: '#7f7f7f', 'ordinary': '#1f77b4', 'indexed': '#d62728', 'union2': '#2ca02c'}


def label(placement, sms, replicas):
    if placement is None:
        return 'unpart.\n(no adapter)'
    if placement == 'ordinary':
        return 'adapter,\nordinary'
    if placement == 'indexed':
        return f'indexed\n{replicas}×{sms} SMs'
    return f'union2\n{replicas}×{2 * sms} SMs'


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = json.loads((root / 'summary' / 'sustained-cells.json').read_text())['rows']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
    for ax, replicas in zip(axes, (2, 3, 4)):
        rate = SAT_RATE[replicas]
        cells = [r for r in rows if r['replicas'] == replicas and r['mps'] and r['graph'] and r['status'].startswith('collected')
                 and r['rate'] == rate and r['runaway_successes'] == 0
                 and (r['placement'] is not None or r['attempt'] in REF_ATTEMPTS[replicas])]
        groups = defaultdict(list)
        for r in cells:
            groups[(r['placement'], r['sms'])].append(r)
        order = sorted(groups, key=lambda k: (['', 'ordinary', 'indexed', 'union2'].index(k[0] or ''), -(k[1] or 0)))
        ref = [r['success_per_s'] for r in groups.get((None, None), [])]
        ref_mean = sum(ref) / len(ref) if ref else None
        for i, k in enumerate(order):
            vals = [r['success_per_s'] for r in groups[k]]
            mean = sum(vals) / len(vals)
            ax.bar(i, mean, color=COLORS[k[0]], alpha=0.75, width=0.7)
            ax.scatter([i] * len(vals), vals, color='black', s=14, zorder=3)
            txt = f'{mean:.1f}'
            if ref_mean and k[0] is not None:
                txt += f'\n{100 * (mean / ref_mean - 1):+.1f} %'
            ax.text(i, max(vals) + 0.25, txt, ha='center', va='bottom', fontsize=8)  # above the highest attempt dot, not the bar
        ax.set_xticks(range(len(order))); ax.set_xticklabels([label(k[0], k[1], replicas) for k in order], fontsize=7.5)
        lo = min(min(r['success_per_s'] for r in groups[k]) for k in order) if order else 0
        ymin = max(0, lo - 4)
        ax.set_ylim(ymin, (max(max(r['success_per_s'] for r in groups[k]) for k in order) + 3.5) if order else 1)
        for i, k in enumerate(order):
            ax.text(i, ymin + 0.25, f'n={len(groups[k])}', ha='center', va='bottom', fontsize=7, color='white')
        ax.set_title(f'DP{replicas} graph + MPS, offered {rate} req/s (saturated)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('completed succ/s in [30,150) s')
    fig.suptitle('Green Context partitions: disjoint SM partitions cost 3–4 % of saturated throughput at every replica count; '
                 'the adapter and overlapping (union2) partitions cost nothing', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'gc-cost-by-dp.{ext}', dpi=150)
    print('->', out / 'gc-cost-by-dp.png')


if __name__ == '__main__':
    main()
