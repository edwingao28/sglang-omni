"""Outcome breakdown of a sustained run's cells from terminal-journal.jsonl; prints non-success/non-rejection records.

  evid_cell_outcomes.py <run-dir> [max_errors]
"""
import collections, glob, json, sys

run, limit = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
for jp in sorted(glob.glob(run + '/cells/*/terminal-journal.jsonl')):
    outs, errs, t0 = collections.Counter(), [], None
    for line in open(jp):
        rec = json.loads(line)
        outs[rec.get('outcome')] += 1
        t0 = t0 or rec.get('send_ns') or rec.get('dispatch_ns')
        if rec.get('outcome') not in ('success', 'rejection', 'admission_rejected', 'rejected'):
            keys = {k: (str(v)[:200]) for k, v in rec.items() if k not in ('result',) and ('err' in k or 'exc' in k or k in ('outcome', 'http_status', 'index'))}
            res = rec.get('result') or {}
            keys['result_error'] = str(res.get('error'))[:200]
            keys['t_s'] = round(((rec.get('send_ns') or rec.get('dispatch_ns') or 0) - t0) / 1e9, 2)
            errs.append(keys)
    print(jp.split('/')[-2], dict(outs))
    for e in errs[:limit]:
        print('   ', e)
