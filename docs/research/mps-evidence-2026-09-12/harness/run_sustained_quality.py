"""Use one owned ASR on this lane for explicit sustained cells after timing has stopped.

Copy of round-1 run_sustained_quality.py with lane CPUs, port and GPU.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from sustained_protocol import (ASR_MODEL, ASR_PORT, BASE, DEPS, RESULT_ROOT, SOURCE, label, lane_cpu_sets,
    lane_gpu_uuid, sha, write_new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', type=label, required=True)
    parser.add_argument('--cells', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    gpu = lane_gpu_uuid()
    cells = [p.resolve() for p in args.cells]
    if len(cells) != len(set(cells)) or any(RESULT_ROOT / 'sustained' not in p.parents for p in cells):
        raise ValueError('Select unique explicit cells in this campaign sustained directory')
    if os.environ.get('SLURM_JOB_ID') != '18012' or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise RuntimeError('Requires recorded job and lane GPU binding')
    if os.environ.get('CAMPAIGN_NSYS') == '1' or os.environ.get('CAMPAIGN_CAPTURE_DIR'):
        raise RuntimeError('ASR must run in a separate unprofiled phase')
    for p in cells:
        if not (p / 'sustained-capture.json').is_file() or (p / 'sustained-quality').exists():
            raise ValueError(f'Missing closed capture or existing quality evidence: {p}')
    before = subprocess.check_output(['nvidia-smi', '--id=' + gpu,
        '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    if before:
        raise RuntimeError('GPU work remains before ASR; finish timing first: ' + before)
    prefix = RESULT_ROOT / ('sustained-quality-' + args.label)
    receipt = prefix.with_suffix('.json')
    asr_log = prefix.with_name(prefix.name + '-asr.log')
    score_log = prefix.with_name(prefix.name + '-score.log')
    lifecycle = prefix.with_name(prefix.name + '-lifecycle.json')
    if any(p.exists() for p in (receipt, asr_log, score_log, lifecycle)):
        raise FileExistsError('Quality outputs must be fresh')
    sets, client_cpus, router_cpus = lane_cpu_sets(1, 8)
    cpus = set(sets[0]) | set(client_cpus) | set(router_cpus)
    os.sched_setaffinity(0, cpus)
    os.environ.update(PYTHONPATH=f'{Path(__file__).resolve().parent}:{SOURCE}:{DEPS}',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    sys.path[:0] = [str(SOURCE), str(DEPS)]
    from benchmarks.benchmarker.utils import managed_omni_server
    state = dict(status='starting', started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cells=list(map(str, cells)), asr_model=ASR_MODEL, asr_port=ASR_PORT, job_id='18012', gpu_uuid=gpu,
        step_id=os.environ.get('SLURM_STEP_ID'), cpu_set=sorted(cpus), gpu_processes_before=before,
        wrapper_sha256=sha(__file__), quality_script_sha256=sha(Path(__file__).with_name('sustained_quality.py')))
    write_new(lifecycle, state)

    def interrupted(sig, frame):
        raise InterruptedError(f'signal {sig}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with managed_omni_server(model_path=ASR_MODEL, port=ASR_PORT, host='127.0.0.1', log_file=asr_log,
                max_running_requests=8, max_queued_requests=16, cuda_graph_max_bs=8, timeout=600,
                wait_for_gpu_release=False):
            command = [sys.executable, str(Path(__file__).with_name('sustained_quality.py')),
                '--cells', *map(str, cells), '--meta', str(BASE / 'inputs/meta.lst'),
                '--asr-port', str(ASR_PORT), '--receipt', str(receipt)]
            state.update(status='scoring', command=command)
            lifecycle.write_text(json.dumps(state, indent=2) + '\n')
            with score_log.open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
            state['score_exit'] = result.returncode
            if result.returncode:
                raise RuntimeError('Occurrence ASR scoring failed; retained its receipt/log')
        state['status'] = 'scoring_completed_owned_asr_stopped'
    except BaseException as error:
        state.update(status='failed_preserved', error=repr(error))
        raise
    finally:
        state['ended_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        lifecycle.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
