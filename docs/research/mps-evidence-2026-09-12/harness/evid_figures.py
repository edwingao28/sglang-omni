"""Figures for the MPS / Green Context evidence campaign (local, matplotlib).

  evid_figures.py <results-root> <out-dir>
<results-root> holds summary/sustained-cells.json (from evid_summarize.py) and analysis/<run_id>/
{trace-analysis,submission-gaps,gpu-metrics}.json (+ .timeseries.csv). Every panel states its
scope in the caption text; no capacity verdicts are drawn by code.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ARM_ORDER = ['DP1-eager-off', 'DP1-eager-off-16c', 'DP2-eager-off', 'DP2-eager-on', 'DP1-graph-off', 'DP2-graph-off',
             'DP2-graph-on', 'DP3-graph-off', 'DP3-graph-on', 'DP4-graph-off', 'DP4-graph-on']
COLORS = {arm: plt.get_cmap('tab20')(i / 20) for i, arm in enumerate(ARM_ORDER)}
PLACEMENT_STYLE = {None: '-', 'ordinary': '-', 'indexed': '--', 'union2': ':'}


def arm_label(row, by_attempt=False):
    label = row['arm'] + (f" {row['attempt']}" if by_attempt else '')
    if row.get('placement'):
        label += f"/{row['placement']}" + (f"-{row['sms']}sm" if row.get('sms') else '')
    return label


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f'{name}.png', dpi=170, facecolor='white')
    fig.savefig(out / f'{name}.pdf', facecolor='white')
    plt.close(fig)
    return name


def group_cells(rows, by_attempt=False):
    groups = {}
    for r in rows:
        groups.setdefault(arm_label(r, by_attempt), {}).setdefault(r['rate'], []).append(r)
    return groups


def sweep_figure(rows, out, arms, name, title, attempt_re=r'.', by_attempt=False, exclude_runaway=True):
    """attempt_re keeps one lane's attempts (Exp 1 = lane B rounds r*/s*, Exp 2 = lane A sweep*); other lanes are never pooled.
    Cells with a runaway success are excluded unless exclude_runaway=False (capacity figures pool runaway-free cells only)."""
    groups = group_cells([r for r in rows if r['arm'] in arms and not r.get('placement') and re.match(attempt_re, r['attempt'])
                          and (not exclude_runaway or not r.get('runaway_successes'))], by_attempt)
    if not groups:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, (label, by_rate) in enumerate(sorted(groups.items(), key=lambda kv: (ARM_ORDER.index(kv[0].split(' ')[0]) if kv[0].split(' ')[0] in ARM_ORDER else 99, kv[0]))):
        rates = sorted(by_rate)
        color = plt.get_cmap('tab10')(i % 10) if by_attempt else COLORS.get(label.split('/')[0], None)
        med = lambda key: [statistics.median(c[key] for c in by_rate[r]) for r in rates]  # noqa: E731
        lo = lambda key: [min(c[key] for c in by_rate[r]) for r in rates]  # noqa: E731
        hi = lambda key: [max(c[key] for c in by_rate[r]) for r in rates]  # noqa: E731
        n = sum(len(v) for v in by_rate.values())
        axes[0].plot(rates, med('success_per_s'), marker='o', color=color, label=f'{label} (n={n})')
        axes[0].fill_between(rates, lo('success_per_s'), hi('success_per_s'), color=color, alpha=.15)
        p95 = [statistics.median(c['latency_p95_s'] for c in by_rate[r] if c['latency_p95_s'] is not None) if any(c['latency_p95_s'] is not None for c in by_rate[r]) else float('nan') for r in rates]
        axes[1].plot(rates, p95, marker='o', color=color, label=label)
        rej = [statistics.median(c['window_rejections'] / max(c['window_sends'], 1) for c in by_rate[r]) for r in rates]
        axes[2].plot(rates, rej, marker='o', color=color, label=label)
    top = max(max(by_rate) for by_rate in groups.values())
    axes[0].plot([0, top], [0, top], color='grey', lw=.8, ls='--', label='offered = completed')
    axes[0].set(xlabel='offered rate (req/s, Poisson)', ylabel='successful completions / s in [30,150) s', title='Sustained completion rate')
    axes[1].set(xlabel='offered rate (req/s)', ylabel='p95 latency of sent-window successes (s)', title='Tail latency (sent cohort, followed through drain)')
    axes[2].set(xlabel='offered rate (req/s)', ylabel='admission rejections / sends in window', title='Rejected share')
    for ax in axes:
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title + ' — median over attempts, band = min..max', fontsize=10)
    return save(fig, out, name)


def gc_figure(rows, out):
    gc = [r for r in rows if r.get('placement')]
    if not gc:
        return None
    groups = group_cells(gc)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for label, by_rate in sorted(groups.items()):
        rates = sorted(by_rate)
        succ = [statistics.median(c['success_per_s'] for c in by_rate[r]) for r in rates]
        p95 = [statistics.median(c['latency_p95_s'] for c in by_rate[r] if c['latency_p95_s'] is not None) for r in rates]
        n = sum(len(v) for v in by_rate.values())
        axes[0].plot(rates, succ, marker='o', label=f'{label} (n={n})')
        axes[1].plot(rates, p95, marker='o', label=label)
    axes[0].set(xlabel='offered rate (req/s)', ylabel='successful completions / s', title='Green Context placements: completion rate')
    axes[1].set(xlabel='offered rate (req/s)', ylabel='p95 latency (s)', title='Green Context placements: tail latency')
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    return save(fig, out, 'gc-placements')


def cpu_figure(rows, out):
    rows = [r for r in rows if (r.get('cpu') or {}).get('roles')]
    if not rows:
        return None
    groups = group_cells(rows)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for label, by_rate in sorted(groups.items(), key=lambda kv: ARM_ORDER.index(kv[0].split('/')[0]) if kv[0].split('/')[0] in ARM_ORDER else 99):
        rates = sorted(by_rate)
        def replica_cores(cell):
            roles = cell['cpu']['roles']
            reps = [v['one_core_pct'] for k, v in roles.items() if k.startswith('replica-') and v['one_core_pct'] is not None]
            return statistics.fmean(reps) / 100 if reps else float('nan')
        ax.plot(rates, [statistics.median(replica_cores(c) for c in by_rate[r]) for r in rates], marker='o',
                ls=PLACEMENT_STYLE.get(by_rate[rates[0]][0].get('placement'), '-'), color=COLORS.get(label.split('/')[0]), label=label)
    ax.set(xlabel='offered rate (req/s)', ylabel='mean cores busy per replica process tree', title='Host CPU per replica (all threads, complete sample intervals)')
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    return save(fig, out, 'cpu-per-replica')


def trace_figure(analysis_root, out):
    records = []
    for d in sorted(analysis_root.glob('*/')):
        ta_path = d / 'trace-analysis.json'
        if not ta_path.exists():
            continue
        ta = json.loads(ta_path.read_text())
        occ = ta['occupancy']
        w = occ['window_ns']
        rec = dict(run_id=ta['run_id'], validity=ta['validity'], union=occ['kernel_union_ns'] / w,
                   overlap=(occ.get('cross_replica_overlap_ns') or 0) / w, zero=occ['zero_kernel_ns'] / w, classes={})
        sg_path = d / 'submission-gaps.json'
        if sg_path.exists():
            sg = json.loads(sg_path.read_text())['submission_gap_analysis']
            rec['classes'] = {k: v * rec['zero'] for k, v in sg['fraction_of_zero_kernel_time_by_class'].items()}
        gm_path = d / 'gpu-metrics.json'
        if gm_path.exists():
            gm = json.loads(gm_path.read_text())
            if gm.get('status') == 'summarized':
                rec['metrics'] = {k.split(' [')[0]: v['mean'] for k, v in gm['metrics'].items()}
        records.append(rec)
    if not records:
        return []
    names = []
    fig, ax = plt.subplots(figsize=(max(8, .9 * len(records)), 4.8))
    x = range(len(records))
    ax.bar(x, [r['union'] - r['overlap'] for r in records], color='#4c72b0', label='kernel present (single owner)')
    ax.bar(x, [r['overlap'] for r in records], bottom=[r['union'] - r['overlap'] for r in records], color='#55a868', label='kernels of ≥2 replicas overlap')
    bottom = [r['union'] for r in records]
    labels = {'before_next_recorded_future_launch_completion': ('host had not finished launching the next kernel', '#c44e52'),
              'recorded_future_kernel_launch_completed': ('launch completed, GPU idle anyway', '#dd8452'),
              'future_kernel_launch_unresolved': ('unresolved', '#8c8c8c'), 'no_future_recorded_kernel_in_window': ('tail', '#cccccc')}
    for key, (label, color) in labels.items():
        vals = [r['classes'].get(key, 0) for r in records]
        if any(vals):
            ax.bar(x, vals, bottom=bottom, color=color, label=f'no kernel: {label}')
            bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([r['run_id'].replace('-trace-', '\n').replace('-n128', '') for r in records], fontsize=6.5)
    ax.set(ylabel='fraction of the measured client window', ylim=(0, 1.02), title='Kernel presence vs. zero-kernel time by cause (Nsight, measured window of 128 requests)')
    ax.legend(fontsize=7, loc='upper right')
    names.append(save(fig, out, 'trace-kernel-presence'))
    metric_records = [r for r in records if r.get('metrics')]
    if metric_records:
        keys = ['SMs Active', 'GR Active', 'SM Issue', 'Tensor Active', 'DRAM Read Bandwidth']
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(metric_records)), 4.4))
        width = .8 / len(keys)
        for i, key in enumerate(keys):
            ax.bar([j + i * width for j in range(len(metric_records))], [r['metrics'].get(key, 0) for r in metric_records], width, label=key)
        ax.set_xticks([j + .4 for j in range(len(metric_records))])
        ax.set_xticklabels([r['run_id'].replace('-trace-', '\n').replace('-n128', '') for r in metric_records], fontsize=7)
        ax.set(ylabel='mean % of peak over the client window (1 kHz samples)', title='Device-wide GPU metrics (nsys --gpu-metrics-set=gh100)')
        ax.grid(alpha=.3, axis='y')
        ax.legend(fontsize=7)
        names.append(save(fig, out, 'trace-gpu-metrics'))
    for d in sorted(analysis_root.glob('*/')):
        csv_path = d / 'gpu-metrics.timeseries.csv'
        if not csv_path.exists():
            continue
        with csv_path.open() as stream:
            reader = csv.reader(stream)
            header = next(reader)
            rows = [row for row in reader]
        if len(header) < 2 or not rows:
            continue
        fig, ax = plt.subplots(figsize=(11, 3.6))
        t = [float(r[0]) for r in rows]
        for col, name in enumerate(header[1:], start=1):
            base = name.split(' [')[0]
            if base in ('SMs Active', 'GR Active', 'SM Issue', 'Tensor Active'):
                ax.plot(t, [float(r[col]) if r[col] else float('nan') for r in rows], lw=.8, label=base)
        ax.set(xlabel='seconds into the measured window', ylabel='% of peak (100 ms bins)', ylim=(0, 100), title=f'{d.name}: GPU activity over the client window')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc='upper right')
        names.append(save(fig, out, f'gpu-timeseries-{d.name}'))
    return names


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / 'summary' / 'sustained-cells.json').read_text())
    rows = summary['rows']
    made = []
    made.append(sweep_figure(rows, out, ['DP1-eager-off', 'DP1-eager-off-16c', 'DP2-eager-off', 'DP2-eager-on', 'DP1-graph-off', 'DP2-graph-off', 'DP2-graph-on'],
                             'exp1-arms', 'Experiment 1: eager/graph × DP1/DP2 × MPS on/off (lane B rounds)', attempt_re=r'^(r\d+|s\d+)$'))
    made.append(sweep_figure(rows, out, ['DP1-graph-off', 'DP2-graph-on', 'DP3-graph-on', 'DP4-graph-on', 'DP2-graph-off', 'DP3-graph-off', 'DP4-graph-off'],
                             'exp2-contention-sweep', 'Experiment 2.1: replicas × MPS contention sweep (Graph arms, lane A; DP1 12/14/18 req/s points from lane C)', attempt_re=r'^(sweep|fine)'))
    made.append(sweep_figure(rows, out, ['DP1-graph-off', 'DP1-eager-off'], 'dp1-cross-gpu', 'DP1 baselines on three GPUs: lane A sweep1 / lane B r1 (runaway service) / lane C xg1', attempt_re=r'^(sweep1|r1|xg1)$', by_attempt=True, exclude_runaway=False))
    made.append(gc_figure(rows, out))
    made.append(cpu_figure(rows, out))
    made.extend(trace_figure(root / 'analysis', out))
    print(json.dumps([m for m in made if m]))


if __name__ == '__main__':
    main()
