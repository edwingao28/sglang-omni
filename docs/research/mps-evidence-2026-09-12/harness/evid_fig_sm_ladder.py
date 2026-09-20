"""SMs Active ladders: 1 kHz device-wide counters (100 ms bins) per topology, one panel per trace.

  evid_fig_sm_ladder.py <results-root> <out-dir>
Grid of panels (SMs Active + GR Active over the whole measured window; shaded band = central half of the
window, dashed = whole-window mean, dotted = mid-window mean) plus a summary row of the means.
Reads analysis/<run_id>/gpu-metrics.timeseries.csv (written by the trace post step).
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

# suffix -> title, [(run_id, topology label, load label)]
LADDERS = {
    '': ('8 req/s, one graph replica vs two (lane B / lane C boots)', [
        ('DP1-eager-off-trace-m1-n128-r8', 'DP1 eager (no CUDA graphs)', '8 req/s'),
        ('DP1-graph-off-trace-m4-n128-r8', 'DP1 CUDA graphs', '8 req/s'),  # m4 = runaway-free instance
        ('DP1-graph-on-trace-m1-n128-r8', 'DP1 CUDA graphs + MPS daemon (control)', '8 req/s'),
        ('DP2-graph-off-trace-m1-n128-r8', 'DP2 graphs, MPS off (time-slice)', '8 req/s'),
        ('DP2-graph-on-trace-m1-n128-r8', 'DP2 graphs, MPS on', '8 req/s'),
    ]),
    '-r24': ('24 req/s: one graph replica saturated vs two', [
        ('DP1-graph-off-trace-m2-n128-r24', 'DP1 CUDA graphs', '24 req/s (saturated)'),
        ('DP1-graph-on-trace-m1-n128-r24', 'DP1 CUDA graphs + MPS daemon (control)', '24 req/s (saturated)'),
        ('DP2-graph-off-trace-m1-n128-r24', 'DP2 graphs, MPS off (time-slice)', '24 req/s (saturated)'),
        ('DP2-graph-on-trace-m1-n128-r24', 'DP2 graphs, MPS on', '24 req/s (keeps up)'),
    ]),
    '-r32': ('DP2 at 32 req/s (both modes saturated)', [
        ('DP2-graph-off-trace-m1-n128-r32', 'DP2 graphs, MPS off (time-slice)', '32 req/s'),
        ('DP2-graph-on-trace-m1-n128-r32', 'DP2 graphs, MPS on', '32 req/s'),
        ('DP2-graph-on-indexed-sms64-trace-m1-n128-r32', 'DP2 + MPS, Green Context 2 × 64 SMs', '32 req/s'),
    ]),
    '-r40': ('DP3 at 40 req/s', [
        ('DP3-graph-off-trace-m1-n128-r40', 'DP3 graphs, MPS off (time-slice)', '40 req/s (saturated)'),
        ('DP3-graph-on-trace-m1-n128-r40', 'DP3 graphs, MPS on', '40 req/s (keeps up)'),
        ('DP3-graph-on-indexed-sms40-trace-m1-n128-r40', 'DP3 + MPS, Green Context 3 × 40 SMs', '40 req/s'),
    ]),
    '-r48': ('DP4 at 48 req/s', [
        ('DP4-graph-off-trace-m1-n128-r48', 'DP4 graphs, MPS off (time-slice)', '48 req/s (saturated)'),
        ('DP4-graph-on-trace-m1-n128-r48', 'DP4 graphs, MPS on', '48 req/s'),
        ('DP4-graph-on-indexed-sms32-trace-m1-n128-r48', 'DP4 + MPS, Green Context 4 × 32 SMs', '48 req/s'),
    ]),
    '-r96': ('saturated windows: 128 requests in a 96 req/s burst, every replica busy through the drain', [
        ('DP2-graph-off-trace-s1-n128-r96', 'DP2, MPS off (time-slice)', '96 req/s burst'),
        ('DP2-graph-on-trace-s1-n128-r96', 'DP2, MPS on', '96 req/s burst'),
        ('DP2-graph-on-indexed-sms64-trace-s1-n128-r96', 'DP2, MPS + Green Context 2 × 64 SMs', '96 req/s burst'),
        ('DP3-graph-off-trace-s1-n128-r96', 'DP3, MPS off (time-slice)', '96 req/s burst'),
        ('DP3-graph-on-trace-s1-n128-r96', 'DP3, MPS on', '96 req/s burst'),
        ('DP3-graph-on-indexed-sms40-trace-s1-n128-r96', 'DP3, MPS + Green Context 3 × 40 SMs', '96 req/s burst'),
        ('DP4-graph-off-trace-s1-n128-r96', 'DP4, MPS off (time-slice)', '96 req/s burst'),
        ('DP4-graph-on-trace-s1-n128-r96', 'DP4, MPS on', '96 req/s burst'),
        ('DP4-graph-on-indexed-sms32-trace-s1-n128-r96', 'DP4, MPS + Green Context 4 × 32 SMs', '96 req/s burst'),
    ]),
}
BLUE, RED = '#1f5fa8', '#c0392b'


def load(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    key = next(k for k in rows[0] if k.startswith('SMs Active'))
    gr = next(k for k in rows[0] if k.startswith('GR Active'))
    t = [float(r['t_offset_s']) for r in rows]
    return t, [float(r[key]) for r in rows], [float(r[gr]) for r in rows]


def ladder(root, out, suffix, title, panels):
    avail = [p for p in panels if (root / 'analysis' / p[0] / 'gpu-metrics.timeseries.csv').exists()]
    if not avail:
        print('no panels available for', suffix or 'r8')
        return
    n = len(avail)
    ncols = 3 if n >= 6 else 2
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(5.6 * ncols, 3.0 * nrows + 3.0), constrained_layout=True)
    gs = gridspec.GridSpec(nrows + 2, ncols, figure=fig, height_ratios=[0.10] + [1.0] * nrows + [0.95])  # row 0 = legend strip
    stats = []
    for i, (rid, topo, load_label) in enumerate(avail):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        t, sm, gr = load(root / 'analysis' / rid / 'gpu-metrics.timeseries.csv')
        t_end = t[-1] if t else 0.0
        mean_sm, mean_gr = sum(sm) / len(sm), sum(gr) / len(gr)
        mid = [v for tt, v in zip(t, sm) if 0.25 * t_end <= tt <= 0.75 * t_end]
        mid_gr = [v for tt, v in zip(t, gr) if 0.25 * t_end <= tt <= 0.75 * t_end]
        mid_sm = sum(mid) / len(mid) if mid else float('nan')
        mid_gr = sum(mid_gr) / len(mid_gr) if mid_gr else float('nan')
        stats.append((topo, mean_sm, mid_sm, mean_gr, mid_gr, rid))
        ax.axvspan(0.25 * t_end, 0.75 * t_end, color='grey', alpha=0.10, lw=0, label='central half of window' if i == 0 else None)
        ax.plot(t, gr, color=RED, lw=0.7, alpha=0.55, label='GR Active (any engine busy)' if i == 0 else None)
        ax.plot(t, sm, color=BLUE, lw=1.1, label='SMs Active' if i == 0 else None)
        ax.axhline(mean_sm, color=BLUE, ls='--', lw=0.9, label='SMs Active, whole-window mean' if i == 0 else None)
        ax.hlines(mid_sm, 0.25 * t_end, 0.75 * t_end, color=BLUE, ls=':', lw=1.8, label='SMs Active, mid-window mean' if i == 0 else None)
        ax.set_xlim(0, t_end)
        ax.set_ylim(0, 100)
        ax.set_title(f'{topo}\n{load_label}, window {t_end:.1f} s', fontsize=10, loc='left', pad=4)
        ax.text(0.02, 0.97, f'SMs Active {mean_sm:.1f} %  ·  mid-window {mid_sm:.0f} %\nGR Active {mean_gr:.0f} %', transform=ax.transAxes,
                fontsize=8.5, va='top', ha='left', color='black', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=0.9))
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8.5)
        if i % ncols == 0:
            ax.set_ylabel('% of peak', fontsize=9)
        if i // ncols == nrows - 1 or i + ncols >= n:
            ax.set_xlabel('seconds into the measured window', fontsize=9)
    # summary row: grouped bars
    axs = fig.add_subplot(gs[nrows + 1, :])
    xs = list(range(n))
    w = 0.2
    axs.bar([x - 1.5 * w for x in xs], [s[1] for s in stats], w, color=BLUE, label='SMs Active, whole window')
    axs.bar([x - 0.5 * w for x in xs], [s[2] for s in stats], w, color=BLUE, alpha=0.45, label='SMs Active, mid-window')
    axs.bar([x + 0.5 * w for x in xs], [s[3] for s in stats], w, color=RED, alpha=0.8, label='GR Active, whole window')
    axs.bar([x + 1.5 * w for x in xs], [s[4] for s in stats], w, color=RED, alpha=0.35, label='GR Active, mid-window')
    for x, s in zip(xs, stats):
        for dx, v in ((-1.5 * w, s[1]), (-0.5 * w, s[2]), (0.5 * w, s[3]), (1.5 * w, s[4])):
            axs.text(x + dx, v + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=7.5)
    axs.set_xticks(xs)
    axs.set_xticklabels([s[0].replace(', ', '\n', 1) for s in stats], fontsize=8.5)
    axs.set_ylim(0, 108)
    axs.set_ylabel('% of peak', fontsize=9)
    axs.grid(alpha=0.25, axis='y')
    axs.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.16), frameon=False)
    axs.set_title('window means', fontsize=10, loc='left', pad=22)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, loc='center', fontsize=8.5, ncol=5, frameon=False)
    fig.suptitle(f'Device-wide SM activity — {title}\n(nsys --gpu-metrics-set=gh100 at 1 kHz, 100 ms bins; one H100; 128-request measured windows)',
                 fontsize=11, x=0.01, ha='left')
    fig.text(0.01, -0.005, 'traces: ' + ', '.join(s[5] for s in stats), fontsize=6.5, color='0.35', va='top')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'sm-active-ladder{suffix}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('panels:', [f'{s[0]}: {s[1]:.1f}/{s[2]:.0f}' for s in stats], '->', out / f'sm-active-ladder{suffix}.png')


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for suffix, (title, panels) in LADDERS.items():
        ladder(root, out, suffix, title, panels)


if __name__ == '__main__':
    main()
