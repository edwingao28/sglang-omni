"""WER by arm / placement over all scored runaway-free sustained cells -> tables/wer-by-arm.md

  evid_wer_table.py <summary-dir>
Pooled WER = total word errors / total reference words across the arm's scored cells.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
rows = json.load((out / 'sustained-cells.json').open())['rows']
by = defaultdict(list)
for r in rows:
    q = r.get('quality')
    if q and r['runaway_successes'] == 0:
        by[(r['arm'], r['placement'] or '—')].append((r['rate'], r['attempt'], 100 * q['corpus_wer'], q['reference_words'], q['errors']))
lines = ['# WER (Qwen3-ASR, corpus level) by arm — scored runaway-free sustained cells', '',
         '| arm | placement | MPS | cells | WER range % | pooled WER % | ref. words | cells (attempt·rate: WER) |', '|---|---|---|---|---|---|---|---|']
for key in sorted(by):
    cells = sorted(by[key]); wers = [c[2] for c in cells]
    err, ref = sum(c[4] for c in cells), sum(c[3] for c in cells)
    mps = 'on' if key[0].endswith('-on') else 'off'
    lines.append(f"| {key[0]} | {key[1]} | {mps} | {len(cells)} | {min(wers):.2f}–{max(wers):.2f} | **{100 * err / ref:.2f}** | {ref} | " + ', '.join(f"{c[1]}·r{c[0]:.0f}: {c[2]:.2f}" for c in cells) + ' |')
n = sum(len(v) for v in by.values()); allw = [c[2] for v in by.values() for c in v]
lines += ['', f'{n} cells, {min(allw):.2f}–{max(allw):.2f} %. The runaway services (§4b) are excluded; their cells score 7 % because of the babbling runaway outputs.']
(out / 'wer-by-arm.md').write_text('\n'.join(lines) + '\n'); print(n, 'cells ->', out / 'wer-by-arm.md')
