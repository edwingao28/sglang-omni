"""Manual Nsight trace of one finite open-loop cell on this lane.

  evid_trace.py --arm DP2-graph-on --attempt t1 --samples 128 --rate 8 [--placement indexed --sms 44] [--gpu-metrics-hz 1000]
Derived from round-1 run-profile-load-v1.py; starts the interactive session, then
launches evid_trace_cell.py inside it and stops after the prepared handshake.
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

from evid_controller import run_identity
from evid_nsys import capture
from sustained_protocol import ARMS, PLACEMENTS, RESULT_ROOT, label, lane_gpu_uuid, rate_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--placement', choices=PLACEMENTS)
    parser.add_argument('--sms', type=int)
    parser.add_argument('--attempt', type=label, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--gpu-metrics-hz', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=1500)
    args = parser.parse_args()
    gpu = lane_gpu_uuid()
    # nsys addresses GPU metrics by system index (PCI order, same as nvidia-smi); never sample GPU 0.
    table = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    index = {row.split(',')[1].strip(): row.split(',')[0].strip() for row in table.strip().splitlines()}[gpu]
    if index == '0':
        raise RuntimeError('GPU index 0 belongs to the other session')
    base_id = run_identity(args.arm, args.placement, args.sms, args.attempt).replace('-sustained-', '-trace-')
    run_id = f'{base_id}-n{args.samples}-r{rate_label(args.rate)}'
    capture_dir = RESULT_ROOT / 'profiles' / run_id
    run_dir = RESULT_ROOT / 'runs' / run_id
    for path in (capture_dir, run_dir):
        if path.exists():
            raise FileExistsError(path)
    env = os.environ.copy()
    env.update(CAMPAIGN_NSYS='1', CAMPAIGN_CAPTURE_DIR=str(capture_dir))
    code = Path(__file__).resolve().parent
    argv = [sys.executable, str(code / 'evid_trace_cell.py'), '--arm', args.arm, '--attempt', args.attempt,
            '--samples', str(args.samples), '--rate', str(args.rate)]
    if args.placement:
        argv += ['--placement', args.placement]
        if args.sms is not None:
            argv += ['--sms', str(args.sms)]
    receipt = dict(run_id=run_id, arm=args.arm, placement=args.placement, sms=args.sms, samples=args.samples,
        request_rate=args.rate, gpu_metrics_hz=args.gpu_metrics_hz, gpu_metrics_device=index if args.gpu_metrics_hz else None,
        gpu_uuid=gpu, cpu_sampler='sample-process-cpu-v3.py',
        profiling_protocol='evid_nsys manual start-before-launch; profile-hook-v4 prepared-stop',
        capture_mode_expected='manual_start_before_launch', preparation_timeout_s=args.timeout,
        analysis_scope='measured arrival/request window only; raw capture includes startup and warmup',
        application_performance_qualified=False, controller_receipt=str(capture_dir / 'profile-receipt.json'),
        argv=argv, status='running')

    def interrupted(sig, frame):
        raise InterruptedError(f'signal {sig}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        controlled = capture(argv, capture_dir, run_dir, run_id, env, timeout=args.timeout,
                             gpu_metrics_hz=args.gpu_metrics_hz, gpu_metrics_device=index)
        for key in ('exit_code', 'export', 'external_stop', 'session_shutdown', 'capture_mode',
                    'cuda_profiler_apis_control_collection', 'error', 'export_error'):
            receipt[key] = controlled.get(key)
        if receipt['capture_mode'] != 'manual_start_before_launch' or receipt['cuda_profiler_apis_control_collection'] is not False:
            raise RuntimeError('Trace did not use the required manual capture mode')
        if controlled['exit_code']:
            raise RuntimeError(f'Trace failed: {controlled.get("error", "controller or export failure")}')
        receipt['status'] = 'captured_coverage_review_pending'
    except BaseException as exc:
        receipt['status'] = 'failed'
        receipt['error'] = repr(exc)
        raise
    finally:
        receipt['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if capture_dir.exists():
            with (capture_dir / 'trace-receipt.json').open('x') as stream:
                stream.write(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({k: receipt.get(k) for k in ('run_id', 'status', 'error', 'exit_code')}), flush=True)


if __name__ == '__main__':
    main()
