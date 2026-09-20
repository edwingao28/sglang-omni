"""Verify per-replica Green Context ownership and TPC-mask disjointness from a real service trace.

  evid_gc_masks.py <capture.sqlite> <run-dir> <trace-analysis.json> <output.json>
For every GPU worker (cuda-started receipt) count measured-window kernels by
(contextId, greenContextId, streamId); join green contexts to
TARGET_INFO_CUDA_CONTEXT_INFO (numMultiprocessors, tpcMask) and compare the
logical TPC masks pairwise across replicas. Ordinary placement must show no green
contexts. This is the empirical gate for "disjoint partitions"; indices alone
never prove it.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3


def tpc_bits(value):
    bits = set()
    for index, word in enumerate(str(value).split(',')):
        parsed = int(word, 0)
        bits.update(index * 32 + bit for bit in range(32) if parsed & (1 << bit))
    return bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run = json.loads((args.run / 'run.json').read_text())
    analysis = json.loads(args.analysis.read_text())
    start, end = analysis['cuda_coverage_checks']['client_window_ns']
    replica_by_pid = {}
    for path in args.run.glob('profiles/events/*/cuda-started-*.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('run_id') == run['run_id']:
            replica_by_pid[receipt['pid']] = path.parent.name
    placement, expected_sms = run.get('placement'), run.get('expected_actual_sms')
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        contexts = {(r['processId'], r['contextId']): dict(r) for r in db.execute('SELECT * FROM TARGET_INFO_CUDA_CONTEXT_INFO')}
        kernels = db.execute('SELECT k.start, k.end, p.pid, k.contextId, k.greenContextId, k.streamId, k.graphNodeId, s.value AS name '
                             'FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN PROCESSES p ON k.globalPid=p.globalPid '
                             'LEFT JOIN StringIds s ON k.shortName=s.id WHERE k.end>? AND k.start<?', (start, end)).fetchall()
    workers, green_rows, issues = [], [], []
    for pid, replica in sorted(replica_by_pid.items(), key=lambda item: item[1]):
        owned = [k for k in kernels if k['pid'] == pid]
        groups = defaultdict(lambda: dict(kernels=0, graph_nodes=0, busy_ns=0, names=Counter()))
        for k in owned:
            key = (k['contextId'], k['greenContextId'], k['streamId'])
            entry = groups[key]
            entry['kernels'] += 1
            entry['graph_nodes'] += bool(k['graphNodeId'])
            entry['busy_ns'] += max(0, min(end, k['end']) - max(start, k['start']))
            entry['names'][k['name']] += 1
        green_ids = sorted({k['greenContextId'] for k in owned if k['greenContextId']})
        green = []
        for gid in green_ids:
            info = contexts.get((pid, gid))
            green_kernels = [k for k in owned if k['greenContextId'] == gid]
            record = dict(green_context_id=gid, kernel_records=len(green_kernels),
                          graph_node_records=sum(bool(k['graphNodeId']) for k in green_kernels),
                          busy_ns=sum(max(0, min(end, k['end']) - max(start, k['start'])) for k in green_kernels),
                          top_names=[name for name, _ in Counter(k['name'] for k in green_kernels).most_common(8)])
            if info:
                record.update(num_multiprocessors=info.get('numMultiprocessors'), num_tpcs=info.get('numTpcs'),
                              tpc_mask=info.get('tpcMask'), device_id=info.get('deviceId'), is_green=info.get('isGreenContext'))
                green_rows.append(dict(pid=pid, replica=replica, gid=gid, bits=tpc_bits(info['tpcMask']) if info.get('tpcMask') else set(),
                                       sms=info.get('numMultiprocessors')))
            else:
                issues.append(f'{replica}: green context {gid} has no TARGET_INFO_CUDA_CONTEXT_INFO row')
            green.append(record)
        total_busy = sum(entry['busy_ns'] for entry in groups.values())
        green_busy = sum(r['busy_ns'] for r in green)
        workers.append(dict(pid=pid, replica=replica, kernel_records=len(owned), green_contexts=green,
            green_busy_fraction_of_owned=green_busy / total_busy if total_busy else None,
            streams=[dict(context_id=key[0], green_context_id=key[1], stream_id=key[2], kernels=v['kernels'], graph_nodes=v['graph_nodes'],
                          busy_ns=v['busy_ns'], top_names=[n for n, _ in v['names'].most_common(5)]) for key, v in sorted(groups.items(), key=lambda kv: -kv[1]['busy_ns'])]))
        if placement in (None, 'ordinary') and green_ids:
            issues.append(f'{replica}: unexpected green contexts {green_ids} under placement {placement}')
        if placement not in (None, 'ordinary'):
            if len(green_ids) != 1:
                issues.append(f'{replica}: expected exactly one green context, saw {green_ids}')
            for record in green:
                if record.get('num_multiprocessors') != expected_sms:
                    issues.append(f'{replica}: green context has {record.get("num_multiprocessors")} SMs, expected {expected_sms}')
                if not record['graph_node_records']:
                    issues.append(f'{replica}: no graph-node kernels on the green context (AR graphs not on partition?)')
    pairs = []
    for i in range(len(green_rows)):
        for j in range(i + 1, len(green_rows)):
            a, b = green_rows[i], green_rows[j]
            if a['pid'] == b['pid']:
                continue
            pairs.append(dict(left=a['replica'], right=b['replica'], shared_tpcs=len(a['bits'] & b['bits']),
                              left_tpcs=len(a['bits']), right_tpcs=len(b['bits'])))
    masks = [r['bits'] for r in green_rows]
    verdict = dict(replicas_with_green=len({r['pid'] for r in green_rows}), pairwise=pairs,
        all_disjoint=bool(pairs) and all(p['shared_tpcs'] == 0 for p in pairs),
        all_identical=bool(masks) and all(m == masks[0] for m in masks),
        union_tpcs=len(set.union(*masks)) if masks else 0)
    if placement == 'indexed' and pairs and not verdict['all_disjoint']:
        issues.append('indexed placement: replica TPC masks overlap')
    if placement == 'union2' and pairs:
        expected_shared = run['sms'] // 2  # two adjacent 40-SM groups share exactly one group = sms SMs = sms/2 TPCs
        bad = [p for p in pairs if p['shared_tpcs'] not in (expected_shared, 0)]
        if bad or not any(p['shared_tpcs'] == expected_shared for p in pairs):
            issues.append(f'union2 placement: pairwise shared TPC counts {[p["shared_tpcs"] for p in pairs]} do not match adjacent-group sharing')
    result = dict(run_id=run['run_id'], placement=placement, sms=run.get('sms'), expected_actual_sms=expected_sms,
        client_window_ns=[start, end], workers=workers, tpc_mask_verdict=verdict, issues=issues,
        status='verified' if not issues else 'issues',
        scope='Real service kernels in the measured window. Green-context ownership is per kernel record; TPC masks are logical per-device masks from CUPTI context info. Vocoder/preprocessing kernels are expected outside the green context.')
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(dict(run_id=run['run_id'], status=result['status'], verdict={k: v for k, v in verdict.items() if k != 'pairwise'}, issues=issues[:5])))


if __name__ == '__main__':
    main()
