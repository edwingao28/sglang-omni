"""Output length of the runaway-prone prompt (pool index 48 = request 48 of every 128-request cycle)
in every sustained cell: per service, the set of successful audio durations per cell.

  evid_runaway_lengths.py <results-root> <out-dir>
Writes runaway-lengths.md. 163.84 s = the 2048-frame talker cap (runaway).
"""
from __future__ import annotations

import collections
import csv
import glob
from pathlib import Path
import sys

PROMPT_INDEX, CYCLE = 48, 128


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    svc = collections.defaultdict(dict)
    variants = []
    for p in sorted(glob.glob(str(root / 'sustained' / '*' / 'cells' / '*' / 'results.csv'))):
        run, cell = p.split('/')[-4], p.split('/')[-2]
        with open(p) as stream:
            rows = [r for r in csv.DictReader(stream) if int(r['id'].rsplit('-', 1)[1]) % CYCLE == PROMPT_INDEX]
        ok = [r for r in rows if r['is_success'] == 'True']
        c = collections.Counter(r['audio_duration_s'] for r in ok)
        svc[run][cell.rsplit('-r', 1)[1]] = (c, len(rows) - len(ok))
        if len(c) > 1:
            main_len = c.most_common(1)[0][0]
            for k, r in enumerate(rows):
                if r['is_success'] == 'True' and r['audio_duration_s'] != main_len:
                    variants.append((cell, r['audio_duration_s'], k, r['id'].rsplit('-', 1)[1], r['latency_s']))
    lines = ['# Output length of the runaway-prone prompt per service and cell',
             '', f'Prompt = pool index {PROMPT_INDEX} (request {PROMPT_INDEX} of every {CYCLE}-request cycle), "Television can make you dumb…". '
             'Cells list successful audio durations (s × count) and admission rejections; 163.84 s = 2048-frame cap (runaway).', '',
             '| service | replicas | cells: duration × count (rej.) |', '|---|---|---|']
    single = collections.Counter(); multi_services = 0; multi_uniform_440 = 0
    for run, cells in sorted(svc.items()):
        replicas = int(run[2]) if run.startswith('DP') else 0
        cell_str = ' '.join(f"r{rate}: " + '/'.join(f'{k}×{v}' for k, v in sorted(c.items())) + (f' ({rej} rej.)' if rej else '')
                            for rate, (c, rej) in sorted(cells.items(), key=lambda kv: float(kv[0])))
        lines.append(f'| `{run}` | {replicas} | {cell_str} |')
        lengths = set(k for c, _ in cells.values() for k in c)
        if replicas == 1:
            single[tuple(sorted(lengths))] += 1
        else:
            multi_services += 1
            multi_uniform_440 += lengths == {'4.4000'}
    lines += ['', '## Summary', '',
              f'- Single-replica services: {sum(single.values())}; length set per service (all cells of a service agree): '
              + ', '.join(f"{'/'.join(k)} × {v}" for k, v in single.most_common()) + '.',
              f'- Multi-replica services: {multi_services}; {multi_uniform_440} produce 4.40 s in every occurrence of every cell; the rest differ in the occurrences listed below (a cell that switched length after a variant shows the new length as its majority):', '']
    lines += ['| cell | variant duration s | occurrence # in cell | request id | latency s |', '|---|---|---|---|---|']
    lines += [f'| `{c}` | {d} | {k} | {i} | {lat} |' for c, d, k, i, lat in variants]
    (out / 'runaway-lengths.md').write_text('\n'.join(lines) + '\n')
    print(f'{len(svc)} services, {len(variants)} variant occurrences -> {out / "runaway-lengths.md"}')


if __name__ == '__main__':
    main()
