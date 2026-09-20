"""Finite-cohort open-loop load under an external Nsight session; lane-parameterised.

Derived from round-1 run-cell-load-profile-v2.py. Any ARMS arm (Eager or Graph,
MPS on/off, DP1..DP4) and optionally a Green Context placement. Started by
evid_trace.py through nsys launch; this process and every child it starts are
inside the traced tree. Handshake with the external controller via
CAMPAIGN_CAPTURE_DIR (capture-finished.json / capture-stop-complete.json).

  evid_trace_cell.py --arm DP2-graph-on --attempt t1 --samples 128 --rate 8 [--placement indexed --sms 44]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from sustained_protocol import (ARMS, BASE, DEPS, INPUT_SHA, PLACEMENTS, REPLICA_PORT, RESULT_ROOT,
    ROUTER_PORT, ROUTER_URL, SOURCE, SOURCE_SHA, NSYS_STOP_ACK_WAIT_S, expected_actual_sms, host_rss_gb, label, wait_ports_free,
    lane_cpu_sets, lane_gpu_uuid, raise_nofile_limit, rate_label, server_config, sha, write_new)
from evid_controller import RamGate, gpu_processes, run_identity, verify_receipt


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--placement', choices=PLACEMENTS)
    parser.add_argument('--sms', type=int)
    parser.add_argument('--attempt', type=label, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--rate', type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.samples <= 128:
        parser.error('--samples must be 1..128 (one pass over the sealed cohort)')
    rate_label(args.rate)
    gpu = lane_gpu_uuid()
    replicas, mps, graph, cores = ARMS[args.arm]
    if args.placement is not None and not graph:
        parser.error('placements require a Graph arm')
    base_id = run_identity(args.arm, args.placement, args.sms, args.attempt).replace('-sustained-', '-trace-')
    run_id = f'{base_id}-n{args.samples}-r{rate_label(args.rate)}'
    out = RESULT_ROOT / 'runs' / run_id
    if out.exists():
        raise FileExistsError(out)
    profile_enabled = os.environ.get('CAMPAIGN_NSYS') == '1'
    if os.environ.get('SLURM_JOB_ID') != '18012' or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise RuntimeError('Expected job 18012 and CUDA_VISIBLE_DEVICES equal to the lane GPU UUID')
    if 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' in os.environ:
        raise RuntimeError('MPS active-thread percentage must remain unset')
    if sha(BASE / 'inputs/meta.lst') != INPUT_SHA:
        raise RuntimeError('Sealed input hash mismatch')
    before = gpu_processes(gpu)
    if before:
        raise RuntimeError('GPU processes remain before boot: ' + before)
    sets, client_cpus, router_cpus = lane_cpu_sets(replicas, cores)
    wait_ports_free([ROUTER_PORT, *[REPLICA_PORT + i for i in range(replicas)]])
    code = Path(__file__).resolve().parent
    from sglang_omni.mps.state import MpsGpuPaths, validate_control_socket
    mps_state_root = Path(f'/tmp/wt-{os.getpid()}')
    validate_control_socket(MpsGpuPaths(mps_state_root, gpu).control_socket)
    # Traced boots also pay for nsys export later; hold the node-wide boot lock until replicas are ready.
    ram_gate = RamGate(replicas, 5.0, run_id=run_id).__enter__()
    rss = ram_gate.sample['host_rss_gb']
    out.mkdir(parents=True, exist_ok=False)
    if args.placement is None:
        configs = [server_config(args.arm)]
        write_new(out / 'config.json', configs[0])
        config_paths = [out / 'config.json'] * replicas
    else:
        configs = [server_config(args.arm, args.placement, args.sms, rank, out / 'resources' / f'replica-{rank}')
                   for rank in range(replicas)]
        config_paths = []
        for rank, value in enumerate(configs):
            write_new(out / f'config-{rank}.json', value)
            config_paths.append(out / f'config-{rank}.json')
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{code}:{SOURCE}:{DEPS}', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', SGLANG_OMNI_MPS_STATE_ROOT=str(mps_state_root),
        TORCHINDUCTOR_CACHE_DIR=f'/work/cache/{BASE.name}/torch', TRITON_CACHE_DIR=f'/work/cache/{BASE.name}/triton',
        CAMPAIGN_MPS_ENABLED='1' if mps else '0', CAMPAIGN_EXPECTED_MPS_CLIENTS=str(replicas) if mps else '0',
        SGLANG_OMNI_STRICT_PORT='1')
    if profile_enabled:
        env['PYTHONPATH'] = f'{BASE}/control/profile-hook-v4:' + env['PYTHONPATH']
        env['SGLANG_OMNI_PROFILE_NVTX'] = '1'
    record = dict(schema_version=2, run_id=run_id, arm=args.arm, attempt=args.attempt, status='starting',
        job_id=os.environ.get('SLURM_JOB_ID'), step_id=os.environ.get('SLURM_STEP_ID'),
        lane=dict(gpu_uuid=gpu, cpus=os.environ.get('LANE_CPUS'), port_base=ROUTER_PORT), gpu_uuid=gpu,
        replicas=replicas, mps=mps, graph=graph, cores_per_replica=cores, cpu_sets=sets,
        router_cpu_set=router_cpus, client_cpu_set=client_cpus, placement=args.placement, sms=args.sms,
        expected_actual_sms=expected_actual_sms(args.placement, args.sms) if args.placement else None,
        config=configs[0], rank_configs=configs if args.placement else None, source=str(SOURCE), source_sha=SOURCE_SHA,
        input_sha256=INPUT_SHA, started_at=now(), processes=[], measured_samples=args.samples, concurrency=0,
        request_rate=args.rate, profile_enabled=profile_enabled, capture_dir=os.environ.get('CAMPAIGN_CAPTURE_DIR'),
        gpu_processes_before=before, host_rss_gb_before=round(rss, 2), cpu_sampler_version='sample-process-cpu-v3.py',
        warmup='upstream runner: 3 concurrent requests of the first sealed input (evid_warmup.py) when profiled, else client --warmup 3',
        load_semantics='finite 1-pass cohort at a fixed Poisson rate (RandomState 42); trace scope only, not capacity')
    processes, logs = [], []
    import resource
    record['child_nofile_soft_limit'] = resource.getrlimit(resource.RLIMIT_NOFILE)[1]

    def save():
        (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')

    def launch(name, argv, cpus):
        command = ['taskset', '-c', ','.join(map(str, cpus)), *argv]
        log = (out / f'{name}.log').open('w')
        logs.append(log)
        p = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=raise_nofile_limit)
        processes.append(p)
        record['processes'].append(dict(name=name, pid=p.pid, command=command))
        save()
        return p

    def ready(port, p, budget=1500):
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            if p.poll() is not None:
                raise RuntimeError(f'Process {p.pid} exited {p.returncode}, port {port}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f'Server {port} readiness exceeded {budget}s')

    def post(port, path, payload):
        request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read() or b'{}')

    def interrupted(sig, frame):
        raise InterruptedError(f'signal {sig}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    cpu_sampler = None
    try:
        save()
        for i, cpus in enumerate(sets):
            entry = ([sys.executable, '-m', 'sglang_omni.cli'] if args.placement is None
                     else [sys.executable, str(code / 'evid_gc_serve.py')])
            p = launch(f'replica-{i}', [*entry, 'serve', '--config', str(config_paths[i]), '--host', '127.0.0.1',
                '--port', str(REPLICA_PORT + i), '--model-name', 'qwen3-tts'], cpus)
            ready(REPLICA_PORT + i, p)
        p = launch('router', [sys.executable, '-m', 'sglang_omni_router.python.serve', '--host', '127.0.0.1',
            '--port', str(ROUTER_PORT), '--worker-urls', *[f'http://127.0.0.1:{REPLICA_PORT + i}' for i in range(replicas)],
            '--policy', 'round_robin', '--model', 'qwen3-tts', '--max-connections', '512', '--max-inflight', '512',
            '--router-state-dir', str(out / 'router-state')], router_cpus)
        ready(ROUTER_PORT, p, 120)
        ram_gate.release()
        record['ram_gate'] = ram_gate.sample
        cpu_sampler = launch('cpu-sampler', [sys.executable, str(BASE / 'control/sample-process-cpu-v3.py'),
            str(out / 'cpu-samples.jsonl'), str(os.getpid())], client_cpus)
        cpu_ready_path = out / 'cpu-sampler-ready.json'
        deadline = time.monotonic() + 60
        while not cpu_ready_path.exists():
            if cpu_sampler.poll() is not None:
                raise RuntimeError('CPU sampler exited before readiness')
            if time.monotonic() > deadline:
                raise TimeoutError('CPU sampler readiness exceeded 60 seconds')
            time.sleep(0.1)
        cpu_ready = json.loads(cpu_ready_path.read_text())
        if cpu_ready['status'] != 'ready' or cpu_ready['sampler_pid'] != cpu_sampler.pid or cpu_ready['mps_enabled'] != mps:
            raise RuntimeError('CPU sampler readiness does not match this cell')
        record['cpu_sampler_ready'] = cpu_ready
        if mps:
            mps_log_dir = Path(cpu_ready['mps_ownership']['pipe']).parent / 'log'
            for name in ('control', 'server'):
                launch(f'logcap-mps-{name}', ['tail', '-n', '+1', '-F', str(mps_log_dir / f'{name}.log')], client_cpus)
        if args.placement is not None:
            client_pids = {e['client_pid'] for e in cpu_ready['mps_ownership']['clients']} if mps else None
            pipe = cpu_ready['mps_ownership']['pipe'] if mps else None
            resources = []
            for rank in range(replicas):
                files = list((out / 'resources' / f'replica-{rank}').glob('qwen-service-resource-*.json'))
                if len(files) != 1:
                    raise RuntimeError('Expected one resource setup receipt per model worker')
                receipt = json.loads(files[0].read_text())
                verify_receipt(receipt, args.placement, args.sms, rank, replicas, gpu, client_pids, pipe)
                resources.append(dict(path=str(files[0]), sha256=sha(files[0]), receipt=receipt))
            record['resource_setup'] = resources
        if profile_enabled:
            warm = launch('warmup', [sys.executable, str(code / 'evid_warmup.py'), str(BASE / 'inputs/meta.lst'),
                                     str(out / 'warmup.json')], client_cpus)
            if warm.wait(timeout=600):
                raise RuntimeError('Profile warmup failed')
            for i in range(replicas):
                event_dir = out / 'profiles/events' / f'replica-{i}'
                event_dir.mkdir(parents=True)
                post(REPLICA_PORT + i, '/start_request_profile', {'run_id': run_id, 'event_dir': str(event_dir)})
                deadline = time.monotonic() + 20
                while not list(event_dir.glob('cuda-started-*.json')):
                    if time.monotonic() > deadline:
                        raise RuntimeError('Worker did not acknowledge CUDA capture start')
                    time.sleep(0.1)
        record.update(status='measuring', ready_at=now(), gpu_processes_ready=gpu_processes(gpu))
        save()
        p = launch('client', [sys.executable, str(code / 'arrival_client.py'), '--model', 'qwen3-tts',
            '--meta', str(BASE / 'inputs/meta.lst'), '--base-url', ROUTER_URL, '--use-existing-server',
            '--generate-only', '--max-samples', str(args.samples), '--warmup', '0' if profile_enabled else '3',
            '--concurrency', '0', '--request-rate', str(args.rate), '--max-new-tokens', '2048', '--seed', '42',
            '--output-dir', str(out / 'metrics'), '--disable-tqdm'], client_cpus)
        deadline = time.monotonic() + 1500
        while p.poll() is None:
            if cpu_sampler.poll() is not None:
                raise RuntimeError('CPU sampler exited during measurement')
            if time.monotonic() > deadline:
                raise TimeoutError('Benchmark exceeded 1500 seconds')
            time.sleep(0.25)
        record['client_exit'] = p.returncode
        if p.returncode:
            raise RuntimeError(f'benchmark exit {p.returncode}')
        if cpu_sampler.poll() is not None:
            raise RuntimeError('CPU sampler exited before measurement completed')
        if profile_enabled:
            for i in range(replicas):
                post(REPLICA_PORT + i, '/stop_request_profile', {'run_id': run_id})
            for i in range(replicas):
                event_dir = out / 'profiles/events' / f'replica-{i}'
                started = {json.loads(f.read_text())['pid'] for f in event_dir.glob('cuda-started-*.json')}
                assert started, 'No GPU worker capture identity'
                deadline = time.monotonic() + 30
                while True:
                    stopped = {json.loads(f.read_text())['pid'] for f in event_dir.glob('cuda-prepared-stop-*.json')}
                    if started <= stopped:
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'Missing CUDA stop acknowledgement: {started - stopped}')
                    time.sleep(0.1)
            capture_dir = Path(os.environ['CAMPAIGN_CAPTURE_DIR'])
            worker_pids = sorted({json.loads(f.read_text())['pid'] for f in (out / 'profiles/events').glob('*/cuda-started-*.json')})
            prepared = {'run_id': run_id, 'worker_pids': worker_pids, 'wall_ns': time.time_ns(),
                        'phase': 'all_workers_synchronized_waiting_external_stop'}
            temporary = capture_dir / 'capture-finished.json.tmp'
            temporary.write_text(json.dumps(prepared))
            temporary.replace(capture_dir / 'capture-finished.json')
            deadline = time.monotonic() + NSYS_STOP_ACK_WAIT_S
            while not (capture_dir / 'capture-stop-complete.json').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('External Nsight controller did not acknowledge stop')
                time.sleep(0.1)
            stopped = json.loads((capture_dir / 'capture-stop-complete.json').read_text())
            record['nsys_external_stop'] = stopped
            if stopped.get('run_id') != run_id or stopped.get('stop_exit') != 0:
                raise RuntimeError('External Nsight stop failed or mismatched; preserve unqualified capture')
        metrics = json.loads((out / 'metrics/speed_results.json').read_text())
        rows = metrics['per_request']
        record['successful_samples'] = sum(bool(r['is_success']) for r in rows)
        record['failed_samples'] = sum(not r['is_success'] for r in rows)
        record['request_identity_complete'] = all(r.get('server_request_id') and r.get('worker_id') for r in rows)
        record['metrics_summary'] = metrics['summary']
        arrivals = json.loads((out / 'metrics/arrivals.json').read_text())
        expected_ids = [line.split('|', 1)[0] for line in (BASE / 'inputs/meta.lst').read_text().splitlines()[:args.samples]]
        schedule, arrival_rows = arrivals['schedule'], arrivals['per_request']
        schedule_hash = hashlib.sha256(json.dumps(schedule, separators=(',', ':')).encode()).hexdigest()
        import numpy as np
        expected_offsets = np.rint(np.cumsum(np.random.RandomState(42).exponential(1.0 / args.rate, size=args.samples)) * 1e9).astype(np.int64).tolist()
        if (arrivals['status'] != 'collected' or arrivals['expected_samples'] != args.samples or arrivals['concurrency'] != 0
                or arrivals['seed'] != 42 or float(arrivals['requested_rate']) != args.rate or len(arrival_rows) != args.samples
                or schedule['sample_ids'] != expected_ids or schedule['offsets_ns'] != expected_offsets
                or arrivals['schedule_sha256'] != schedule_hash or [row['sample_id'] for row in arrival_rows] != expected_ids
                or [row['index'] for row in arrival_rows] != list(range(args.samples))
                or [row['nominal_offset_ns'] for row in arrival_rows] != expected_offsets):
            raise RuntimeError('Arrival cohort, seed, rate, order, offsets, or schedule hash does not match the cell')
        by_id = {row['id']: row for row in rows}
        if set(by_id) != set(expected_ids) or len(by_id) != args.samples:
            raise RuntimeError('Metric cohort differs from sealed arrival cohort')
        for row in arrival_rows:
            metric = by_id[row['sample_id']]
            if (not row.get('is_success') or row.get('server_request_id') != metric['server_request_id']
                    or row.get('worker_id') != metric['worker_id'] or row.get('send_ns') != metric['request_start_ns']
                    or row.get('end_ns') != metric['request_end_ns']):
                raise RuntimeError('Arrival request identity or timing does not match metrics')
        record['arrival_schedule_sha256'] = schedule_hash
        record['arrival_validation'] = 'complete cohort and exact seeded schedule, identity, and client timing verified'
        if len(rows) != args.samples or record['failed_samples'] or not record['request_identity_complete']:
            raise RuntimeError('Incomplete requests or request/worker identity; retain invalid run')
        record['status'] = 'collected'
    except BaseException as e:
        record['status'] = 'failed'
        record['error'] = repr(e)
        raise
    finally:
        ram_gate.close()
        for p in reversed(processes):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        record['ended_at'] = now()
        save()
        for log in logs:
            log.close()
        print(json.dumps({'run_id': run_id, 'status': record['status'], 'error': record.get('error')}), flush=True)


if __name__ == '__main__':
    main()
