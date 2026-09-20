"""One resident service per boot, explicit Poisson rate cells; lane-parameterised.

Derived from round-1 sustained_controller.py and sustained_gc_controller.py.
  python evid_controller.py --arm DP2-graph-on --attempt r1 --rates 8 40
  python evid_controller.py --arm DP3-graph-on --placement indexed --sms 44 --attempt gc1 --rates 24 40
The lane (GPU UUID, CPUs, port base) comes from LANE_* environment variables. No
allocation or container is created here; GPU 0 and CPUs 0-31 are refused.
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
import time
import urllib.request

from sustained_protocol import (ARMS, BASE, DEPS, INPUT_SHA, PLACEMENTS, REPLICA_PORT, RESULT_ROOT,
    ROUTER_PORT, ROUTER_URL, SAMPLER_SHA, SOURCE, SOURCE_SHA, expected_actual_sms, host_rss_gb, cgroup_anon_gb, cgroup_limit_gb, RAM_BOOT_LOCK,
    label, lane_cpu_sets, lane_gpu_uuid, raise_nofile_limit, rate_label, server_config, sha, wait_ports_free, write_new)

RAM_BUDGET_GB = 56.0
PER_REPLICA_GB = 4.5  # measured: ~12 GB anon for three DP1 services incl. router/client/telemetry
RAM_LEDGER = BASE / 'locks' / 'ram-ledger'  # NFS: visible to every lane container regardless of namespaces
RAM_WAIT_SECONDS = 2400
CONTROL_FILES = ('sustained_protocol.py', 'sustained_client.py', 'evid_controller.py', 'sustained_analyzer.py',
    'analyze_load_cell.py', 'evid_gc_adapter.py', 'evid_gc_serve.py', 'qwen_gc.py', 'gc_resources.py',
    'evid_gc_configs/__init__.py', 'evid_gc_configs/qwen/__init__.py', 'evid_gc_configs/qwen/config.py')
RUNTIME_FILES = ('benchmarks/tasks/tts.py', 'benchmarks/eval/benchmark_tts_seedtts.py',
    'benchmarks/benchmarker/runner.py', 'benchmarks/benchmarker/data.py',
    'sglang_omni_router/python/proxy.py', 'sglang_omni_router/python/data_plane.py',
    'sglang_omni/models/qwen3_tts/engine_builder.py', 'sglang_omni/models/qwen3_tts/model_runner.py')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_identity(arm, placement, sms, attempt):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    label(attempt)
    parts = [arm]
    if placement is not None:
        if placement not in PLACEMENTS:
            raise ValueError('Unknown placement')
        parts.append(placement)
        if placement != 'ordinary':
            if type(sms) is not int or sms <= 0:
                raise ValueError('indexed/union2 need --sms')
            parts.append(f'sms{sms}')
    return '-'.join(parts + ['sustained', attempt])


def paths_for(run_id, rates, root=RESULT_ROOT):
    names = [rate_label(r) for r in rates]
    if not names or len(names) != len(set(names)):
        raise ValueError('Rates must be nonempty and unique; explicit order is preserved')
    out = root / 'sustained' / run_id
    if out.exists():
        raise FileExistsError(out)
    return out, [f'{run_id}-r{name}' for name in names]


def gpu_processes(gpu):
    return subprocess.check_output(['nvidia-smi', '--id=' + gpu,
        '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()


def preflight(gpu, replicas, log=print):
    if os.environ.get('SLURM_JOB_ID') != '18012' or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise RuntimeError('Expected job 18012 and CUDA_VISIBLE_DEVICES equal to the lane GPU UUID')
    if os.environ.get('CAMPAIGN_NSYS') == '1' or os.environ.get('SGLANG_OMNI_PROFILE_NVTX') == '1':
        raise RuntimeError('Sustained performance requires an unprofiled service')
    if 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' in os.environ:
        raise RuntimeError('MPS active-thread percentage must remain unset')
    if sha(BASE / 'inputs/meta.lst') != INPUT_SHA or sha(BASE / 'control/sample-process-cpu-v3.py') != SAMPLER_SHA:
        raise RuntimeError('Sealed input or original CPU sampler hash mismatch')
    before = gpu_processes(gpu)
    if before:
        raise RuntimeError('GPU processes remain before resident boot; inspect ownership: ' + before)
    return before


class RamGate:
    """Node-wide boot serialisation: hold /tmp lock from the RSS sample until every replica is ready.

    Host RSS (all PIDs, no PID namespace) is the anonymous-memory proxy; the cgroup anon counter is
    recorded alongside and, when readable, also gated against the cgroup limit minus a margin.
    """

    def __init__(self, replicas, extra_gb, log=print, run_id=None):
        self.need = PER_REPLICA_GB * replicas + extra_gb
        self.log = log
        self.handle = None
        self.sample = None
        self.run_id = run_id
        self.lane = os.environ.get('LANE_GPU_UUID', 'unknown-lane')
        self.reservation = RAM_LEDGER / f'{self.lane}.json'

    def others_reserved_gb(self):
        """Sum of other lanes' live reservations (files older than 6 h count as stale)."""
        total, entries = 0.0, []
        RAM_LEDGER.mkdir(parents=True, exist_ok=True)
        for path in RAM_LEDGER.glob('*.json'):
            if path == self.reservation:
                continue
            try:
                entry = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if time.time() - entry.get('at_unix', 0) > 6 * 3600:
                continue
            total += float(entry.get('need_gb', 0))
            entries.append(dict(lane=path.stem, need_gb=entry.get('need_gb'), run_id=entry.get('run_id')))
        return total, entries

    def reserve(self):
        RAM_LEDGER.mkdir(parents=True, exist_ok=True)
        tmp = self.reservation.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(lane=self.lane, need_gb=self.need, run_id=self.run_id, pid=os.getpid(), at_unix=time.time())))
        tmp.replace(self.reservation)

    def close(self):
        """Drop the reservation when the service is torn down (call from the run's finally)."""
        self.release()
        try:
            self.reservation.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self):
        import fcntl
        self.handle = open(RAM_BOOT_LOCK, 'a+')
        waited = time.monotonic()
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        deadline = time.monotonic() + RAM_WAIT_SECONDS
        while True:
            rss, anon, limit = host_rss_gb(), cgroup_anon_gb(), cgroup_limit_gb()
            others, entries = self.others_reserved_gb()
            # /proc and the cgroup files inside a lane container only cover that container, so the
            # cross-lane budget is enforced through the NFS ledger; the local samples are recorded.
            cg_ok = anon is None or limit is None or anon + self.need <= limit - 4.0
            if others + self.need <= RAM_BUDGET_GB and rss + self.need <= RAM_BUDGET_GB and cg_ok:
                self.reserve()
                self.sample = dict(host_rss_gb=round(rss, 2), cgroup_anon_gb=None if anon is None else round(anon, 2),
                                   cgroup_limit_gb=limit, need_gb=self.need, others_reserved_gb=others, other_lanes=entries,
                                   lock_wait_s=round(time.monotonic() - waited, 1))
                return self
            self.log(json.dumps({'ram_wait': True, 'host_rss_gb': round(rss, 1), 'cgroup_anon_gb': anon, 'need_gb': self.need,
                                 'others_reserved_gb': others, 'budget_gb': RAM_BUDGET_GB}), flush=True)
            if time.monotonic() > deadline:
                self.release()
                raise RuntimeError(f'Host RSS {rss:.1f} GB (cgroup anon {anon}) leaves no room for {self.need:.0f} GB')
            time.sleep(30)

    def release(self):
        if self.handle is not None:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __exit__(self, *exc):
        self.release()


def verify_receipt(receipt, placement, sms, rank, replicas, gpu, client_pids, mps_pipe):
    expected = expected_actual_sms(placement, sms)
    resource = receipt['resource']
    if (receipt.get('status') != 'stream_and_capture_setup_verified' or receipt.get('rank') != rank
            or receipt.get('placement') != placement or resource.get('pid') != receipt.get('pid')
            or receipt.get('expected_actual_sms') != expected or resource.get('actual_sms') != expected
            or receipt.get('group_count') != replicas
            or receipt.get('cuda_mps_active_thread_percentage') is not None
            or receipt.get('cuda_visible_devices') != gpu or resource.get('cuda_visible_devices') != gpu):
        raise RuntimeError(f'Resource receipt does not match rank {rank}/{placement}/{expected} SMs')
    if client_pids is not None and receipt['pid'] not in client_pids:
        raise RuntimeError('Resource receipt PID is not an owned MPS client')
    if mps_pipe is not None and (receipt.get('cuda_mps_pipe_directory') != mps_pipe
                                or resource.get('cuda_mps_pipe_directory') != mps_pipe):
        raise RuntimeError('Resource stream does not belong to this lane MPS pipe')
    if placement == 'ordinary':
        if resource.get('requested_sms') is not None or receipt.get('requested_sms') is not None:
            raise RuntimeError('Ordinary placement must not request SMs')
    else:
        wanted = [rank] if placement == 'indexed' else [rank, (rank + 1) % replicas]
        if (resource.get('requested_sms') != sms or resource.get('group_index') != rank
                or resource.get('group_count') != replicas or resource.get('selected_group_indices') != wanted
                or resource.get('union_next') != (placement == 'union2')
                or any(count != sms for count in resource.get('group_sms', []))):
            raise RuntimeError('Actual resource split selection differs from the requested group(s)')
    after = receipt['resource_after_graph_setup']
    if (set(after) != {'stream_handle', 'stream_id', 'context_handle', 'actual_sms'}
            or any(resource.get(key) != value for key, value in after.items())):
        raise RuntimeError('Resource stream changed after Graph setup')
    captures = receipt['captures']
    if (any(captures.get(key) != resource.get('stream_handle') for key in
            ('generation_named_stream_handle', 'predictor_stream_handle', 'decode_capture_stream_handle'))
            or not captures.get('predictor_graph_count') or not captures.get('decode_capture_batch_sizes')):
        raise RuntimeError('Capture stream does not match actual resource stream')


def execute(arm, placement, sms, attempt, rates):
    gpu = lane_gpu_uuid()
    run_id = run_identity(arm, placement, sms, attempt)
    out, cell_ids = paths_for(run_id, rates)
    replicas, mps, graph, cores = ARMS[arm]
    if placement is not None and not graph:
        raise ValueError('Resource placements require a Graph arm')
    before = preflight(gpu, replicas)
    sets, client_cpus, router_cpus = lane_cpu_sets(replicas, cores)
    wait_ports_free([ROUTER_PORT, *[REPLICA_PORT + i for i in range(replicas)]])
    # Keep the UNIX socket independent of a descriptive, potentially long run ID.
    from sglang_omni.mps.state import MpsGpuPaths, validate_control_socket
    mps_state_root = Path(f'/tmp/we-{os.getpid()}')
    validate_control_socket(MpsGpuPaths(mps_state_root, gpu).control_socket)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'run.json', dict(run_id=run_id, status='waiting_ram', started_at=now()))
    try:
        ram_gate = RamGate(replicas, 2.0, run_id=run_id).__enter__()
    except Exception as error:
        (out / 'run.json').write_text(json.dumps(dict(run_id=run_id, status='failed_ram_wait', error=repr(error), ended_at=now()), indent=2))
        raise
    rss = ram_gate.sample['host_rss_gb']
    (out / 'run.json').unlink()
    (out / 'cells').mkdir()
    code = Path(__file__).resolve().parent
    if placement is None:
        configs = [server_config(arm)]
        write_new(out / 'config.json', configs[0])
        config_paths = [out / 'config.json'] * replicas
    else:
        configs = [server_config(arm, placement, sms, rank, out / 'resources' / f'replica-{rank}') for rank in range(replicas)]
        config_paths = []
        for rank, value in enumerate(configs):
            write_new(out / f'config-{rank}.json', value)
            config_paths.append(out / f'config-{rank}.json')
    env = os.environ.copy()
    for name in ('CAMPAIGN_NSYS', 'CAMPAIGN_CAPTURE_DIR', 'SGLANG_OMNI_PROFILE_NVTX'):
        env.pop(name, None)
    env.update(PYTHONPATH=f'{code}:{SOURCE}:{DEPS}', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        SGLANG_OMNI_MPS_STATE_ROOT=str(mps_state_root),
        TORCHINDUCTOR_CACHE_DIR=f'/work/cache/{BASE.name}/torch', TRITON_CACHE_DIR=f'/work/cache/{BASE.name}/triton',
        CAMPAIGN_MPS_ENABLED='1' if mps else '0', CAMPAIGN_EXPECTED_MPS_CLIENTS=str(replicas) if mps else '0',
        SGLANG_OMNI_STRICT_PORT='1')
    state = dict(schema_version=2, run_id=run_id, arm=arm, attempt=attempt, status='starting', started_at=now(),
        job_id=os.environ['SLURM_JOB_ID'], step_id=os.environ.get('SLURM_STEP_ID'), node=os.environ.get('SLURMD_NODENAME'),
        lane=dict(gpu_uuid=gpu, cpus=os.environ.get('LANE_CPUS'), port_base=ROUTER_PORT),
        gpu_uuid=gpu, replicas=replicas, mps=mps, graph=graph, cores_per_replica=cores,
        cpu_sets=sets, client_cpu_set=client_cpus, router_cpu_set=router_cpus,
        placement=placement, sms=sms, expected_actual_sms=expected_actual_sms(placement, sms) if placement else None,
        config=configs[0], rank_configs=configs if placement else None,
        source=str(SOURCE), source_sha=SOURCE_SHA, deps=str(DEPS), input_sha256=INPUT_SHA, sampler_sha256=SAMPLER_SHA,
        gpu_processes_before=before, host_rss_gb_before=round(rss, 2),
        control_sha256={name: sha(code / name) for name in CONTROL_FILES},
        runtime_source_sha256={name: sha(SOURCE / name) for name in RUNTIME_FILES},
        mps_active_thread_percentage={'present': False}, mps_state_root=str(mps_state_root),
        selected_rates=rates, cells=[], processes=[],
        preparation='Three concurrent first-input warmups followed by three full 128-input cycles, C16; unique occurrence IDs. Once per resident boot.',
        scope='150s Poisson arrivals per rate (RandomState 42); first 30s settling, fixed [30,150) counts. Cells drain before next rate; no automatic capacity verdict.')
    write_new(out / 'run.json', state)
    processes, logs = [], []
    telemetry = sampler = None
    import resource
    state['child_nofile_soft_limit'] = resource.getrlimit(resource.RLIMIT_NOFILE)[1]

    def save():
        (out / 'run.json').write_text(json.dumps(state, indent=2) + '\n')

    def launch(name, args, cpus):
        log_path = out / f'{name}-{len(processes):02d}.log'
        stream = log_path.open('x')
        logs.append(stream)
        command = ['taskset', '-c', ','.join(map(str, cpus)), *args]
        p = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=raise_nofile_limit)
        processes.append(p)
        state['processes'].append(dict(name=name, pid=p.pid, command=command, log=str(log_path)))
        save()
        return p

    def alive(exclude=None):
        for p in processes:
            record = next(r for r in state['processes'] if r['pid'] == p.pid)
            if p is not exclude and record['name'] != 'client' and p.poll() is not None:
                raise RuntimeError(f'Owned service/telemetry/sampler exited: {record["name"]} {p.returncode}')

    def ready(port, budget=1500):
        end = time.monotonic() + budget
        while time.monotonic() < end:
            alive()
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f'Server {port} readiness exceeded {budget}s')

    def client(output, prefix, extra, budget):
        p = launch('client', [sys.executable, str(code / 'sustained_client.py'), '--output', str(output),
            '--meta', str(BASE / 'inputs/meta.lst'), '--prefix', prefix, *extra], client_cpus)
        end = time.monotonic() + budget
        while p.poll() is None:
            alive(exclude=p)
            if time.monotonic() >= end:
                raise TimeoutError('Sustained client exceeded finite lifecycle budget')
            time.sleep(.25)
        alive(exclude=p)
        if p.returncode:
            raise RuntimeError(f'Client exited {p.returncode}; preserve phase and stop')

    def idle_snapshot_valid(snapshot):
        servers, router = snapshot['servers'], snapshot['router']
        return (len(servers) == replicas and len(router.get('workers', [])) == replicas
            and all(s.get('running') is True and s.get('total_requests') == 0 and s.get('pending_completions') == 0
                    and not any(s.get('request_states', {}).values()) for s in servers)
            and all(w.get('active_requests') == 0 and w.get('routable') is True for w in router['workers']))

    def verify_drained():
        # Only used after all HTTP outcomes succeeded or were known rejections.
        deadline, snapshots, consecutive = time.monotonic() + 30, [], 0
        while time.monotonic() < deadline:
            alive()
            snapshot = {'at': now(), 'servers': []}
            for i in range(replicas):
                with urllib.request.urlopen(f'http://127.0.0.1:{REPLICA_PORT + i}/health', timeout=2) as response:
                    snapshot['servers'].append(json.load(response))
            with urllib.request.urlopen(f'{ROUTER_URL}/workers', timeout=2) as response:
                snapshot['router'] = json.load(response)
            snapshots.append(snapshot)
            consecutive = consecutive + 1 if idle_snapshot_valid(snapshot) else 0
            if consecutive == 2:
                return snapshots
            time.sleep(1)
        state['failed_drain_snapshots'] = snapshots
        raise RuntimeError('Server coordinator and router did not both reach zero twice; stop resident reuse')

    try:
        telemetry = launch('gpu-telemetry', ['nvidia-smi', '--id=' + gpu,
            '--query-gpu=timestamp,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used',
            '--format=csv', '--loop=1'], client_cpus)
        state['telemetry_path'] = state['processes'][-1]['log']
        for i, cpus in enumerate(sets):
            entry = ([sys.executable, '-m', 'sglang_omni.cli'] if placement is None
                     else [sys.executable, str(code / 'evid_gc_serve.py')])
            launch(f'replica-{i}', [*entry, 'serve', '--config', str(config_paths[i]), '--host', '127.0.0.1',
                '--port', str(REPLICA_PORT + i), '--model-name', 'qwen3-tts'], cpus)
            ready(REPLICA_PORT + i)
        launch('router', [sys.executable, '-m', 'sglang_omni_router.python.serve', '--host', '127.0.0.1',
            '--port', str(ROUTER_PORT), '--worker-urls', *[f'http://127.0.0.1:{REPLICA_PORT + i}' for i in range(replicas)],
            '--policy', 'round_robin', '--model', 'qwen3-tts', '--max-connections', '512', '--max-inflight', '512',
            '--router-state-dir', str(out / 'router-state')], router_cpus)
        ready(ROUTER_PORT, 120)
        sampler = launch('cpu-sampler', [sys.executable, str(BASE / 'control/sample-process-cpu-v3.py'),
            str(out / 'cpu-samples.jsonl'), str(os.getpid())], client_cpus)
        marker = out / 'cpu-sampler-ready.json'
        deadline = time.monotonic() + 60
        while not marker.exists():
            alive()
            if time.monotonic() > deadline:
                raise TimeoutError('Original CPU sampler readiness exceeded 60s')
            time.sleep(.1)
        proof = json.loads(marker.read_text())
        if proof['status'] != 'ready' or proof['sampler_pid'] != sampler.pid or proof['mps_enabled'] != mps:
            raise RuntimeError('Sampler ownership/readiness does not match resident service')
        if placement is not None:
            client_pids = {e['client_pid'] for e in proof['mps_ownership']['clients']} if mps else None
            pipe = proof['mps_ownership']['pipe'] if mps else None
            resources = []
            for rank in range(replicas):
                files = list((out / 'resources' / f'replica-{rank}').glob('qwen-service-resource-*.json'))
                if len(files) != 1:
                    raise RuntimeError('Expected one resource setup receipt per model worker')
                receipt = json.loads(files[0].read_text())
                verify_receipt(receipt, placement, sms, rank, replicas, gpu, client_pids, pipe)
                resources.append(dict(path=str(files[0]), sha256=sha(files[0]), receipt=receipt))
            if mps and {e['receipt']['pid'] for e in resources} != client_pids:
                raise RuntimeError('Resource workers differ from the exact owned MPS clients')
            state.update(resource_setup=resources, actual_model_kernel_ownership='pending external trace audit',
                globally_disjoint_replica_sms_proven=False)
        if mps:
            mps_log_dir = Path(proof['mps_ownership']['pipe']).parent / 'log'
            for name in ('control', 'server'):
                launch(f'logcap-mps-{name}', ['tail', '-n', '+1', '-F', str(mps_log_dir / f'{name}.log')], client_cpus)
        ram_gate.release()
        state.update(cpu_sampler_ready=proof, ready_at=now(), status='conditioning', ram_gate=ram_gate.sample,
            gpu_processes_ready=gpu_processes(gpu), host_rss_gb_ready=round(host_rss_gb(), 2),
            cgroup_anon_gb_ready=cgroup_anon_gb())
        save()
        client(out / 'preparation', run_id, ['--prepare'], 1500)
        preparation = json.loads((out / 'preparation/sustained-capture.json').read_text())
        if preparation['status'] != 'prepared' or len(preparation['per_request']) != 387:
            raise RuntimeError('Warmup / conditioning incomplete')
        state['post_conditioning_drain'] = verify_drained()
        state.update(status='measuring', preparation_capture_sha256=sha(out / 'preparation/sustained-capture.json'))
        save()
        from sustained_analyzer import analyze_capture
        for rate, cell_id in zip(rates, cell_ids):
            row = dict(cell_id=cell_id, rate=rate, started_at=now(), status='running')
            state['cells'].append(row)
            save()
            output = out / 'cells' / cell_id
            client(output, cell_id, ['--rate', str(rate)], 540)
            capture = json.loads((output / 'sustained-capture.json').read_text())
            manifest = json.loads((output / 'occurrence-manifest.json').read_text())
            analysis = analyze_capture(capture, manifest)
            if capture['manifest_sha256'] != sha(output / 'occurrence-manifest.json'):
                analysis['issues'].append('Manifest byte hash differs')
            write_new(output / 'flow-analysis.json', analysis)
            row.update(status='collected', ended_at=now(), outcomes=analysis['outcomes'], window=analysis['window'],
                flow_issues=analysis['issues'], capture_sha256=sha(output / 'sustained-capture.json'),
                manifest_sha256=sha(output / 'occurrence-manifest.json'))
            save()
            print(json.dumps({'checkpoint': row, 'sent_window': analysis['sent_window_cohort']}), flush=True)
            # Admission rejection is a recorded load outcome. Other failures may leave
            # server work alive; do not contaminate the next resident cell or retry.
            # Client-side dispatch shortfall (arrival not sent because the client's event loop lagged past the
            # grace) is recorded and tolerated below 1 % of arrivals; server-side failures still stop the sweep.
            client_side = {k: analysis['outcomes'].get(k, 0) for k in ('not_dispatched', 'not_sent_before_cutoff')}
            row['client_dispatch_shortfall'] = client_side
            unsafe = {k: v for k, v in analysis['outcomes'].items() if k not in ('success', 'admission_rejection', *client_side) and v}
            if sum(client_side.values()) > 0.01 * max(1, sum(analysis['outcomes'].values())):
                unsafe.update({k: v for k, v in client_side.items() if v})
            if analysis['issues'] or unsafe:
                row['status'] = 'collected_with_failure_stop'
                save()
                raise RuntimeError(f'Preserved incomplete protocol/service outcomes; stop next rate: {unsafe}, {analysis["issues"]}')
            row['post_cell_drain'] = verify_drained()
            save()
        state['status'] = 'selected_cells_collected_review_and_quality_pending'
    except BaseException as exc:
        state.update(status='failed_preserved', error=repr(exc))
        save()
        raise
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        state.update(ended_at=now(), process_exit_codes={str(p.pid): p.returncode for p in processes})
        try:
            state['gpu_processes_after'] = gpu_processes(gpu)
        except Exception as error:  # noqa: BLE001 - record, never mask the primary outcome
            state['gpu_processes_after'] = repr(error)
        ram_gate.close()
        save()
        for stream in logs:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--placement', choices=PLACEMENTS)
    parser.add_argument('--sms', type=int)
    parser.add_argument('--attempt', type=label, required=True)
    parser.add_argument('--rates', type=float, nargs='+', required=True)
    args = parser.parse_args()
    if args.placement in (None, 'ordinary') and args.sms is not None:
        parser.error('--sms only applies to indexed/union2 placements')

    def interrupted(sig, frame):
        raise InterruptedError(f'signal {sig}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(args.arm, args.placement, args.sms, args.attempt, args.rates)


if __name__ == '__main__':
    main()
