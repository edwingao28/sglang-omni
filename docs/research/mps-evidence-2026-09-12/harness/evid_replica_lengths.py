"""Per-replica output length of the runaway-prone prompt (pool index 48) in multi-replica sustained services.

  evid_replica_lengths.py <results-root> <out-dir>
Joins each cell's terminal-journal.jsonl (occurrence index, client_request_uuid, audio_duration_s) with the
service's router log (`route_completed request_id=<uuid> worker=127.0.0.1:<port>`) and tabulates, per service and
worker, the durations the prompt returned. Writes replica-lengths.md + replica-lengths.json.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path
import re
import sys

CYCLE, PROMPT_INDEX, CAP_S = 128, 48, 160.0
ROUTE_RE = re.compile(r'route_completed request_id=([0-9a-f-]{36}) worker=127\.0\.0\.1:(\d+)')


def fmt(durs):
    c = collections.Counter(durs)
    return '/'.join(('cap' if d >= CAP_S else f'{d:.2f}') + f'×{n}' for d, n in sorted(c.items(), key=lambda x: -x[1]))


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    services = []
    for run_dir in sorted((root / 'sustained').glob('*/')):
        routers = sorted(run_dir.glob('router-*.log'))
        if not routers:
            continue
        log = routers[0].read_text(errors='replace')
        m = re.search(r'workers=(\d+)', log)
        replicas = int(m.group(1)) if m else None
        if not replicas or replicas < 2:
            continue
        uuid_to_port = dict(ROUTE_RE.findall(log))
        per_worker = collections.defaultdict(lambda: collections.defaultdict(list))  # port -> cell -> durations
        unmatched = 0
        for jp in sorted(run_dir.glob('cells/*/terminal-journal.jsonl')):
            cell = jp.parent.name.rsplit('-', 1)[1]
            with jp.open() as stream:
                for line in stream:
                    rec = json.loads(line)
                    if rec.get('index', -1) % CYCLE != PROMPT_INDEX or rec.get('outcome') != 'success':
                        continue
                    port = uuid_to_port.get(rec.get('client_request_uuid'))
                    dur = float(rec['result']['audio_duration_s'])
                    if port is None:
                        unmatched += 1
                        continue
                    per_worker[port][cell].append(dur)
        workers = []
        for port in sorted(per_worker):
            durs = [d for cell in per_worker[port].values() for d in cell]
            majority = collections.Counter(durs).most_common(1)[0][0]
            workers.append(dict(port=int(port), n=len(durs), majority_s=majority, runaway=sum(d >= CAP_S for d in durs),
                                cells={c: sorted(v) for c, v in per_worker[port].items()}))
        services.append(dict(service=run_dir.name, replicas=replicas, workers_probed=len(workers), unmatched=unmatched, workers=workers))
    (out / 'replica-lengths.json').write_text(json.dumps(services, indent=1) + '\n')
    lines = ['# Per-replica output length of the runaway-prone prompt (multi-replica sustained services)', '',
             f'Prompt = pool index {PROMPT_INDEX} ("Television can make you dumb…"). Each successful occurrence is joined to the replica that served it '
             'through the client request uuid in `terminal-journal.jsonl` and the router\'s `route_completed` line. "cap" = 163.84 s (2048 frames, runaway). '
             'Per worker: durations × count, per cell (rate). Round-robin routing with a 128-request cycle sends every occurrence in a cell to one worker '
             'unless admission rejections shift the alignment.', '',
             '| service | replicas | worker | n | majority s | per cell |', '|---|---|---|---|---|---|']
    probed = off = ran = 0
    for s in services:
        for w in s['workers']:
            probed += 1
            off += w['majority_s'] != 4.40 and w['majority_s'] < CAP_S
            ran += w['majority_s'] >= CAP_S
            cells = ' '.join(f"{c}: {fmt(v)}" for c, v in sorted(w['cells'].items(), key=lambda x: int(x[0][1:])))
            flag = ' **' if w['majority_s'] != 4.40 else ''
            maj = 'cap' if w['majority_s'] >= CAP_S else f"{w['majority_s']:.2f}"
            lines.append(f"| `{s['service']}` | {s['replicas']} | {w['port']} | {w['n']} | {maj}{flag} | {cells} |")
    boots = sum(s['replicas'] for s in services)
    unprobed = boots - probed
    lines += ['', '## Summary', '',
              f'- {len(services)} multi-replica services, {boots} replica boots; {probed} replicas received at least one occurrence of the prompt '
              f'({unprobed} never did — the cycle alignment kept the prompt on the other replicas).',
              f'- Majority length 4.40 s: {probed - off - ran} replicas; another fixed length: {off}; the cap (runaway replica): {ran}.',
              f'- Unmatched successes (uuid absent from router log): {sum(s["unmatched"] for s in services)}.']
    (out / 'replica-lengths.md').write_text('\n'.join(lines) + '\n')
    print(f'{len(services)} services, {probed} probed replicas of {boots}; off-4.40: {off}; runaway: {ran} -> {out}')


if __name__ == '__main__':
    main()
