"""Per-replica latency and admission rejections per saturated cell (multi-replica sustained services).

  evid_replica_latency.py <results-root> <out-dir> [min_rate]
Joins terminal-journal.jsonl (client_request_uuid, outcome, latency) with the router's route_completed lines
(uuid -> worker port) and writes replica-latency.md: per cell, each worker's success count, p50 latency and rejections.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path
import re
import statistics
import sys

ROUTE_RE = re.compile(r'route_completed request_id=([0-9a-f-]{36}) worker=127\.0\.0\.1:(\d+)')


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    min_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# Per-replica latency at saturation (multi-replica sustained services, cells at ≥ %g req/s)' % min_rate, '',
             'Each successful request is joined to the replica that served it (client uuid → router `route_completed`). '
             'Per worker: successes / p50 latency s / admission rejections (503, replica queue full). The router is round-robin, so '
             'arrival shares are equal; unequal p50 means unequal service share.', '',
             '| service | rate | succ/s | per worker: n / p50 s / rej | p50 spread (max−min s) |', '|---|---|---|---|---|']
    records = []
    for run_dir in sorted((root / 'sustained').glob('*/')):
        routers = sorted(run_dir.glob('router-*.log'))
        if not routers:
            continue
        u2p = dict(ROUTE_RE.findall(routers[0].read_text(errors='replace')))
        for jp in sorted(run_dir.glob('cells/*/terminal-journal.jsonl'), key=lambda p: float(p.parent.name.rsplit('-r', 1)[1])):
            rate = float(jp.parent.name.rsplit('-r', 1)[1])
            if rate < min_rate:
                continue
            lat, rej, succ = collections.defaultdict(list), collections.Counter(), 0
            with jp.open() as stream:
                for line in stream:
                    rec = json.loads(line)
                    port = u2p.get(rec.get('client_request_uuid'))
                    if not port:
                        continue
                    if rec.get('outcome') == 'success':
                        lat[port].append(rec['result']['latency_s']); succ += 1
                    elif rec.get('outcome') == 'admission_rejection':
                        rej[port] += 1
            if len(lat) < 2:
                continue
            p50 = {p: statistics.median(v) for p, v in lat.items()}
            cell_json = jp.parent / 'flow-analysis.json'
            sps = ''
            if cell_json.exists():
                fa = json.loads(cell_json.read_text())
                w = fa.get('window') or {}
                if w.get('success_per_s') is not None:
                    sps = f"{w['success_per_s']:.2f}"
            per = ' '.join(f"{p[-1]}: {len(lat[p])} / {p50[p]:.2f} / {rej.get(p, 0)}" for p in sorted(lat))
            spread = max(p50.values()) - min(p50.values())
            records.append(dict(service=run_dir.name, rate=rate, per_worker={p: dict(n=len(lat[p]), p50_s=p50[p], rejections=rej.get(p, 0)) for p in sorted(lat)}, p50_spread_s=spread))
            lines.append(f"| `{run_dir.name}` | {rate:g} | {sps} | {per} | {spread:.2f} |")
    (out / 'replica-latency.json').write_text(json.dumps(records, indent=1) + '\n')
    (out / 'replica-latency.md').write_text('\n'.join(lines) + '\n')
    print(f'{len(records)} cells -> {out / "replica-latency.md"}')


if __name__ == '__main__':
    main()
