"""One figure: completed throughput and latency vs offered rate for DP1-DP4, MPS on/off.

  evid_fig_rate_curves.py <evidence-root>

Reads tables/sustained-cells.json (276 cells, no new runs) and writes
figures/rate-curves-mps.{png,pdf} + tables/rate-curves-mps.md.

Per curve it marks two rates that the campaign already defines in RESULTS.md Sec. 2:
  keeps-up ceiling = highest measured offered rate where, in EVERY collected cell at
                     that rate, the pending queue does not grow and no request is
                     rejected inside the [30,150) s window.
  saturation       = plateau of median completed succ/s over the rates above it.
Hue = topology, line style = MPS. Colors are the validated 4-slot categorical palette.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HUE = {'DP1': '#2a78d6', 'DP2': '#eb6834', 'DP3': '#1baf7a', 'DP4': '#eda100'}
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'
# DP1 has no lane-A MPS-on sweep; its pair is the matched lane-C boots c1-c5.
LANE = {'DP1-graph-on': ('c',), 'DP1-graph-off': ('sweep', 'fine1', 'c')}
ARMS = [(f'DP{n}-graph-{m}', n, m) for n in (1, 2, 3, 4) for m in ('off', 'on')]


def load(root):
    rows = json.loads((root / 'tables' / 'sustained-cells.json').read_text())['rows']
    keep, drop = {}, 0
    for r in rows:
        if r.get('placement') or r['status'] != 'collected':
            continue
        if r.get('runaway_successes'):           # capacity/latency reads use runaway-free cells
            drop += 1
            continue
        if not any(r['arm'] == a for a, _, _ in ARMS):
            continue
        if not r['attempt'].startswith(LANE.get(r['arm'], ('sweep',))):
            continue
        keep.setdefault(r['arm'], {}).setdefault(r['rate'], []).append(r)
    return keep, drop


def cell_keeps_up(c):
    """No rejections, completions track arrivals, and the backlog does not build up.

    pending_change is a raw in-flight count at the window edges and is noisy by a few
    requests, so it is read against the window's own send count rather than against 0."""
    return (c['window_rejections'] == 0
            and c['success_per_s'] >= 0.98 * c['offered_per_s']
            and c['pending_change'] <= 0.02 * c['window_sends'])


def keeps_up_to(by_rate):
    """Highest measured rate at which every collected cell keeps up."""
    ok = [x for x in sorted(by_rate) if all(cell_keeps_up(c) for c in by_rate[x])]
    return max(ok) if ok else None


def plateau(by_rate):
    med = {r: st.median(c['success_per_s'] for c in by_rate[r]) for r in by_rate}
    top = max(med.values())
    flat = sorted(r for r in med if med[r] >= 0.97 * top)
    return st.median(med[r] for r in flat), min(flat), top


def main(root):
    root = Path(root).expanduser()
    data, dropped = load(root)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor=SURFACE)
    lines, notes = [], []
    for arm, dp, mps in ARMS:
        by_rate = data.get(arm)
        if not by_rate:
            continue
        rates = sorted(by_rate)
        color, on = HUE[f'DP{dp}'], mps == 'on'
        style = dict(color=color, lw=2.0, ls='-' if on else (0, (5, 2.5)),
                     marker='o' if on else 's', ms=6.5, mfc=color if on else SURFACE, mew=1.6)
        med = lambda k: [st.median(c[k] for c in by_rate[r]) for r in rates]  # noqa: E731
        lo = lambda k: [min(c[k] for c in by_rate[r]) for r in rates]         # noqa: E731
        hi = lambda k: [max(c[k] for c in by_rate[r]) for r in rates]         # noqa: E731
        n = sum(len(v) for v in by_rate.values())
        lane = {'DP1-graph-on': 'lane C', 'DP1-graph-off': 'lane A+C'}.get(arm, '')
        label = f"DP{dp} · MPS {mps}" + (f" ({lane})" if lane else ' (lane A)') + f"  n={n}"
        for i, key in enumerate(('success_per_s', 'latency_p50_s', 'latency_p95_s')):
            ln, = ax[i].plot(rates, med(key), **style, label=label if i == 0 else None, zorder=3)
            ax[i].fill_between(rates, lo(key), hi(key), color=color, alpha=.12, lw=0, zorder=1)
            if i == 0:
                lines.append(ln)
        ku, (plat, plat_from, _) = keeps_up_to(by_rate), plateau(by_rate)
        ax[0].axhline(plat, color=color, lw=.9, ls=':', alpha=.55, zorder=0)
        if ku is not None:
            y = st.median(c['success_per_s'] for c in by_rate[ku])
            ax[0].plot([ku], [y], marker='o', ms=14, mfc='none', mec=color, mew=2.0, zorder=4)
            for i, key in ((1, 'latency_p50_s'), (2, 'latency_p95_s')):
                ax[i].plot([ku], [st.median(c[key] for c in by_rate[ku])], marker='o', ms=14,
                           mfc='none', mec=color, mew=2.0, zorder=4)
        notes.append((f'DP{dp}', mps, lane or 'lane A', n, ku, plat, plat_from,
                      st.median(c['latency_p50_s'] for c in by_rate[max(rates)]),
                      st.median(c['latency_p95_s'] for c in by_rate[max(rates)]),
                      st.median(c['latency_p50_s'] for c in by_rate[ku]) if ku else None,
                      st.median(c['latency_p95_s'] for c in by_rate[ku]) if ku else None))

    top = max(max(v) for v in data.values())
    ax[0].plot([0, top], [0, top], color='#a9a8a3', lw=1.0, ls='--', zorder=0,
               label='offered = completed')
    ax[0].set(xlabel='offered rate (req/s, open-loop Poisson)',
              ylabel='completed successes / s', title='Completed throughput')
    ax[1].set(xlabel='offered rate (req/s)', ylabel='latency p50 (s)', title='Median latency')
    ax[2].set(xlabel='offered rate (req/s)', ylabel='latency p95 (s)', title='Tail latency')
    for a in ax[1:]:
        a.set_yscale('log')
    for a in ax:
        a.grid(alpha=.22, lw=.7)
        a.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            a.spines[s].set_color('#d5d4cf')
        a.tick_params(colors=INK2, labelsize=9)
        a.xaxis.label.set_color(INK2)
        a.yaxis.label.set_color(INK2)
        a.title.set_color(INK)
    ax[0].legend(fontsize=8, loc='upper left', frameon=False, labelcolor=INK2, ncol=1)
    ax[0].annotate('◯ = keeps-up ceiling\n⋯ = saturated plateau', xy=(.98, .04),
                   xycoords='axes fraction', ha='right', fontsize=8.5, color=INK2)
    fig.suptitle('Qwen3-Omni on one H100 — replicas × MPS, sustained [30,150) s window; '
                 'line = median over boots, band = min..max', fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .95))
    figs = root / 'figures'
    for ext in ('png', 'pdf'):
        fig.savefig(figs / f'rate-curves-mps.{ext}', dpi=170, facecolor=SURFACE)
    plt.close(fig)

    out = ['# Rate curves: DP1–DP4 × MPS on/off (derived from tables/sustained-cells.json)', '',
           f'Runaway-containing cells excluded ({dropped}). "keeps up" = in every cell at that '
           'rate: no rejections, completions >= 98 % of arrivals, backlog growth < 2 % of sends. Saturation = median completed '
           'succ/s over rates within 3 % of the arm maximum.', '',
           '| topology | MPS | lane | cells | keeps up to | p50 / p95 there (s) | saturates at '
           '(succ/s) | from rate | p50 / p95 at 48 req/s (s) |', '|---|---|---|---|---|---|---|---|---|']
    for dp, mps, lane, n, ku, plat, pf, p50e, p95e, p50k, p95k in notes:
        out.append(f'| {dp} | {mps} | {lane} | {n} | {f"{ku:g} req/s" if ku else "—"} | '
                   f'{f"{p50k:.2f} / {p95k:.2f}" if p50k else "—"} | **{plat:.1f}** | {pf:g} req/s | '
                   f'{p50e:.2f} / {p95e:.2f} |')
    (root / 'tables' / 'rate-curves-mps.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '~/benchmark-results/mps-evidence-20260912-a01')
