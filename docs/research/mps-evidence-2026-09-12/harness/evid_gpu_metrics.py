"""Crop Nsight GPU-metrics samples to the measured client window and summarize.

  evid_gpu_metrics.py <capture.sqlite> <trace-analysis.json> <output.json> [--bin-ms 100]
Uses the client window (trace clock) from analyze_trace output. SM Active etc. are
device-wide sampled counters (percent of peak per sampling period), independent of
kernel-presence unions. Writes summary JSON and a binned time-series CSV next to it.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import statistics

# nsys names carry a unit suffix, e.g. 'SMs Active [Throughput %]'; match on the base name and prefer the
# throughput-percent variant so the time series is one comparable column per counter.
INTEREST = ('SMs Active', 'SM Issue', 'Tensor Active', 'GR Active', 'DRAM Read Bandwidth', 'DRAM Write Bandwidth',
            'Compute Warps in Flight', 'Unallocated Warps in Active SMs', 'PCIe RX Throughput', 'PCIe TX Throughput',
            'GPC Clock Frequency')


def canonical(name):
    base, _, unit = name.partition(' [')
    return base, unit.rstrip(']')


def pick_series(series):
    chosen = {}
    for name in series:
        base, unit = canonical(name)
        if base not in INTEREST:
            continue
        current = chosen.get(base)
        if current is None or (unit == 'Throughput %' and canonical(current)[1] != 'Throughput %'):
            chosen[base] = name
    return chosen


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * q
    lo, hi = int(at), min(int(at) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--bin-ms', type=float, default=100.0)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    start, end = analysis['cuda_coverage_checks']['client_window_ns']
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'GPU_METRICS' not in tables or 'TARGET_INFO_GPU_METRICS' not in tables:
            result = dict(run_id=analysis['run_id'], status='no_gpu_metrics_tables', tables=sorted(t for t in tables if 'GPU' in t))
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result))
            return
        info_cols = [row[1] for row in db.execute('PRAGMA table_info(TARGET_INFO_GPU_METRICS)')]
        metric_cols = [row[1] for row in db.execute('PRAGMA table_info(GPU_METRICS)')]
        names = {}
        for row in db.execute('SELECT * FROM TARGET_INFO_GPU_METRICS'):
            record = dict(zip(info_cols, row))
            names[(record.get('typeId'), record.get('metricId'))] = record.get('metricName')
        rows = db.execute('SELECT timestamp, typeId, metricId, value FROM GPU_METRICS WHERE timestamp>=? AND timestamp<? ORDER BY timestamp',
                          (start, end)).fetchall()
        full_bounds = db.execute('SELECT min(timestamp), max(timestamp), count(*) FROM GPU_METRICS').fetchone()
    series = {}
    for ts, type_id, metric_id, value in rows:
        name = names.get((type_id, metric_id), f'{type_id}:{metric_id}')
        series.setdefault(name, []).append((ts, value))
    summary = {}
    for name, points in series.items():
        values = [v for _, v in points]
        spacing = [b[0] - a[0] for a, b in zip(points, points[1:])]
        summary[name] = dict(samples=len(values), mean=statistics.fmean(values), p50=percentile(values, .5),
            p95=percentile(values, .95), max=max(values), min=min(values),
            zero_fraction=sum(v == 0 for v in values) / len(values),
            below_5pct_fraction=sum(v < 5 for v in values) / len(values) if 'Active' in name or 'Issue' in name else None,
            sample_spacing_ns_median=percentile(spacing, .5) if spacing else None)
    bins = {}
    width = int(args.bin_ms * 1e6)
    chosen = pick_series(series)
    present = [base for base in INTEREST if base in chosen]
    for base in present:
        for ts, value in series[chosen[base]]:
            bins.setdefault((ts - start) // width, {}).setdefault(base, []).append(value)
    csv_path = args.output.with_suffix('.timeseries.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t_offset_s', *[chosen[base] for base in present]])
        for index in sorted(bins):
            writer.writerow([round(index * args.bin_ms / 1000, 3), *[
                round(statistics.fmean(bins[index][base]), 3) if bins[index].get(base) else '' for base in present]])
    result = dict(run_id=analysis['run_id'], status='summarized', client_window_ns=[start, end], window_s=(end - start) / 1e9,
        gpu_metrics_full_capture=dict(first_ns=full_bounds[0], last_ns=full_bounds[1], rows=full_bounds[2]),
        gpu_metrics_columns=metric_cols, metrics=dict(sorted(summary.items())), timeseries_csv=str(csv_path), bin_ms=args.bin_ms,
        scope='Device-wide sampled hardware counters (percent of peak per sampling interval) cropped to the client window; not per-process. Kernel union from analyze_trace is presence, this is utilization.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    result['selected_series'] = chosen
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({base: (round(summary[name]['mean'], 2), round(summary[name]['p95'], 2), round(summary[name]['zero_fraction'], 3)) for base, name in chosen.items()}))


if __name__ == '__main__':
    main()
