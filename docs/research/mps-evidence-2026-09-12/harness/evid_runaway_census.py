"""Census of the runaway prompt across every service instance that received it.

  evid_runaway_census.py <results-root> [prompt-id]
Scans runs/<run>/metrics/results.csv (traces) and sustained/<run>/cells/<cell>/**/results.csv (sustained cells);
a sustained run = one service instance (all its cells share the boot), a trace run = one instance.
Prints per-instance: arm, instance, cells where the prompt was sent, ran away (audio >= 160 s) or not.
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1]); pid = sys.argv[2] if len(sys.argv) > 2 else 'common_voice_en_17275818-common_voice_en_17275816'
inst = defaultdict(lambda: dict(sent=0, runaway=0, cells=set()))
for p in sorted(root.glob('runs/*/metrics/results.csv')) + sorted(root.glob('sustained/*/cells/*/**/results.csv')):
    run = p.relative_to(root).parts[1]
    cell = p.relative_to(root).parts[3] if p.relative_to(root).parts[0] == 'sustained' else 'trace'
    for x in csv.DictReader(p.open()):
        if x.get('id') != pid or not x.get('audio_duration_s'):
            continue
        d = inst[run]; d['sent'] += 1; d['cells'].add(cell.split('-r')[-1] if cell != 'trace' else 'trace')
        d['runaway'] += float(x['audio_duration_s']) >= 160
by_arm = defaultdict(lambda: [0, 0])
rows = []
for run, d in sorted(inst.items()):
    arm = run.split('-sustained-')[0].split('-trace-')[0]
    ran = d['runaway'] > 0
    by_arm[arm][0] += 1; by_arm[arm][1] += ran
    rows.append((arm, run, d['sent'], d['runaway'], ','.join(sorted(d['cells']))))
    print(f"{arm:20s} {run:45s} sent {d['sent']:3d} runaway {d['runaway']:3d} cells {rows[-1][4]}")
print('\nper arm: instances that received the prompt -> instances where it ran away')
for arm, (n, k) in sorted(by_arm.items()):
    print(f'  {arm:20s} {k}/{n}')
