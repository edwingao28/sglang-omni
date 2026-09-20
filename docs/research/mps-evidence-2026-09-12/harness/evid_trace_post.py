"""Post-process one closed trace on the node: coverage analysis, submission gaps, GPU metrics, GC masks.

  evid_trace_post.py <run_id>
Reads RESULT_ROOT/profiles/<run_id>/capture.sqlite and RESULT_ROOT/runs/<run_id>;
writes RESULT_ROOT/analysis/<run_id>/{trace-analysis,submission-gaps,gpu-metrics,gc-masks}.json.
Never opens the GPU. Safe to queue between lane cells.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from sustained_protocol import RESULT_ROOT


def main():
    run_id = sys.argv[1]
    code = Path(__file__).resolve().parent
    database = RESULT_ROOT / 'profiles' / run_id / 'capture.sqlite'
    run = RESULT_ROOT / 'runs' / run_id
    out = RESULT_ROOT / 'analysis' / run_id
    out.mkdir(parents=True, exist_ok=True)
    if not database.is_file():
        raise FileNotFoundError(database)
    steps = [
        ('trace-analysis', [sys.executable, str(code / 'analyze_trace.py'), str(database), str(run), str(out / 'trace-analysis.json')]),
        ('submission-gaps', [sys.executable, str(code / 'analyze_submission_gaps.py'), str(database), str(run),
                             str(out / 'trace-analysis.json'), str(out / 'submission-gaps.json')]),
        ('gpu-metrics', [sys.executable, str(code / 'evid_gpu_metrics.py'), str(database), str(out / 'trace-analysis.json'), str(out / 'gpu-metrics.json')]),
        ('gc-masks', [sys.executable, str(code / 'evid_gc_masks.py'), str(database), str(run), str(out / 'trace-analysis.json'), str(out / 'gc-masks.json')]),
    ]
    status = {}
    for name, argv in steps:
        with (out / f'{name}.log').open('w') as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        status[name] = result.returncode
        print(json.dumps({'run_id': run_id, 'step': name, 'exit': result.returncode}), flush=True)
        if name == 'trace-analysis' and result.returncode:
            break
    (out / 'post-status.json').write_text(json.dumps(status, indent=2) + '\n')
    raise SystemExit(0 if all(code == 0 for code in status.values()) else 1)


if __name__ == '__main__':
    main()
