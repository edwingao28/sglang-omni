"""Manual interactive Nsight collection with the prepared-stop handshake (lane copy).

Derived from round-1 nsys_support_v5.py. start -> launch -> stop per the Nsight
Systems user guide; capture-range=none so CUDA profiler APIs never gate
collection. Optional GPU-metrics sampling (SM Active, SM Issue, DRAM BW) on the
CUDA-visible device only. Raw capture includes startup and warmup; analyze only
the measured request window.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from nsys_support_v3 import NSYS, export_report, process_identity, save
from sustained_protocol import NSYS_STOP_TIMEOUT_S


def commands(argv, output, session, gpu_metrics_hz=0, gpu_metrics_device='cuda-visible'):
    output = Path(output)
    start = [NSYS, 'start', '--session-new=' + session, '--capture-range=none',
             '--sample=none', '--cpuctxsw=none', '--discard-environment=true',
             '--force-overwrite=false', '--output=' + str(output / 'capture')]
    if gpu_metrics_hz:
        start += [f'--gpu-metrics-devices={gpu_metrics_device}', f'--gpu-metrics-frequency={int(gpu_metrics_hz)}',
                  '--gpu-metrics-set=gh100']
    launch = [NSYS, 'launch', '--session=' + session, '--trace=cuda,nvtx',
              '--cuda-graph-trace=node', '--cuda-event-trace=false',
              '--cuda-memory-usage=false', '--trace-fork-before-exec=false',
              '--inherit-environment=true', '--show-output=true', '--wait=all',
              sys.executable, '-S', '-B', str(Path(__file__).with_name('server_entry.py')),
              str(output / 'server-identity.json'), *map(str, argv)]
    return {'start': start, 'launch': launch,
            'stop': [NSYS, 'stop', '--session=' + session],
            'status': [NSYS, 'status', '--session=' + session],
            'shutdown': [NSYS, 'shutdown', '--session=' + session, '--kill=none'],
            'sessions': [NSYS, 'sessions', 'list']}


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def same_live_process(identity):
    try:
        current = process_identity(identity['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return False
    return current['start_ticks'] == identity['start_ticks'] and current['state'] != 'Z'


def validate_prepared(finished, run, run_id):
    if (not isinstance(finished, dict) or finished.get('run_id') != run_id or
            type(finished.get('wall_ns')) is not int or
            finished.get('phase') != 'all_workers_synchronized_waiting_external_stop'):
        raise ValueError('Wrong capture-finished identity or phase')
    pids = finished.get('worker_pids', [])
    if not pids or any(type(pid) is not int or pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
        raise ValueError('Capture-finished needs unique worker PIDs')
    started = [read_json(path) for path in Path(run).glob('profiles/events/*/cuda-started-*.json')]
    prepared = [read_json(path) for path in Path(run).glob('profiles/events/*/cuda-prepared-stop-*.json')]
    if (any(not item or item.get('run_id') != run_id for item in started + prepared) or
            len(started) != len(pids) or len(prepared) != len(pids) or
            {item['pid'] for item in started} != set(pids) or
            {item['pid'] for item in prepared} != set(pids)):
        raise ValueError('Worker start/prepared acknowledgements do not match the cohort')
    start_times = {item['pid']: item['wall_ns'] for item in started}
    for item in prepared:
        if (item.get('synchronized') is not True or item.get('cuda_profiler_stop_called') is not False or
                type(item.get('wall_ns')) is not int or
                not start_times[item['pid']] <= item['wall_ns'] <= finished['wall_ns']):
            raise ValueError('Worker has not confirmed synchronized preparation without CUDA stop')
    return prepared


def capture(argv, output, run, run_id, env, timeout=900, gpu_metrics_hz=0, gpu_metrics_device='cuda-visible'):
    output, run = Path(output), Path(run)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError('Nsight output must be fresh: ' + str(output))
    session = 'evid-' + run_id
    plan = commands(argv, output, session, gpu_metrics_hz, gpu_metrics_device)
    receipt = {'run_id': run_id, 'session': session, 'controller_pid': os.getpid(),
               'created_ns': time.time_ns(), 'exit_code': 1, 'commands': plan, 'steps': [],
               'capture_mode': 'manual_start_before_launch', 'gpu_metrics_hz': gpu_metrics_hz,
               'gpu_metrics_device': gpu_metrics_device if gpu_metrics_hz else None,
               'cuda_profiler_apis_control_collection': False,
               'raw_scope': 'target startup, warmup and measured requests through external stop',
               'analysis_scope': 'crop to measured request timestamps; exclude startup and warmup',
               'preparation_timeout_s': timeout}
    save(output / 'command.json', receipt)
    launch = None
    created = False
    stop_attempted = False

    def invoke(name, command, seconds):
        step = {'name': name, 'argv': command, 'started_ns': time.time_ns()}
        receipt['steps'].append(step)
        try:
            with (output / (name + '.log')).open('xb') as log:
                proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=seconds)
            step['returncode'] = proc.returncode
        except Exception as error:
            step.update(returncode=None, error=repr(error))
        step['finished_ns'] = time.time_ns()
        save(output / 'profile-receipt.json', receipt)
        return step

    def stop(prepared_ok):
        nonlocal stop_attempted
        stop_attempted = True
        invoke('sessions-before-stop', plan['sessions'], 10)
        invoke('status-before-stop', plan['status'], 10)
        result = invoke('stop', plan['stop'], NSYS_STOP_TIMEOUT_S)
        report = output / 'capture.nsys-rep'
        closed = result['returncode'] == 0 and report.is_file() and report.stat().st_size > 0
        acknowledgement = {'run_id': run_id, 'session': session, 'wall_ns': time.time_ns(),
                           'stop_exit': 0 if prepared_ok and closed else 1,
                           'nsys_stop_return': result['returncode'], 'report_closed': closed,
                           'workers_prepared': prepared_ok, 'controller_pid': os.getpid()}
        receipt['external_stop'] = acknowledgement
        save(output / 'capture-stop-complete.json', acknowledgement)
        invoke('status-after-stop', plan['status'], 10)
        return acknowledgement

    def wait_cell(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            record = read_json(run / 'run.json')
            identity = read_json(output / 'server-identity.json')
            if record and record.get('ended_at') and identity and not same_live_process(identity):
                receipt['cell'] = record
                return record
            if launch is not None and launch.poll() is not None and not record:
                raise RuntimeError(f'Cell exited {launch.returncode} before writing run.json (preflight failure)')
            time.sleep(.2)
        raise TimeoutError('Recorded cell did not complete its own cleanup')

    with (output / 'nsys.log').open('xb') as log:
        try:
            result = invoke('start', plan['start'], 60)
            if result['returncode'] != 0:
                raise RuntimeError('Interactive Nsight start failed')
            created = True
            launch = subprocess.Popen(plan['launch'], env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt['launch_pid'] = launch.pid
            deadline = time.monotonic() + timeout
            while not (output / 'capture-finished.json').exists():
                code = launch.poll()
                if code not in (None, 0):
                    raise RuntimeError('Nsight launch exited ' + str(code))
                record = read_json(run / 'run.json')
                if record and (record.get('status') == 'failed' or record.get('ended_at')):
                    raise RuntimeError('Cell ended before all workers were prepared')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Capture preparation exceeded deadline')
                time.sleep(.2)
            finished = read_json(output / 'capture-finished.json')
            receipt['prepared_workers'] = validate_prepared(finished, run, run_id)
            receipt['workers_at_stop'] = [process_identity(pid) for pid in finished['worker_pids']]
            if any(item['state'] == 'Z' for item in receipt['workers_at_stop']):
                raise RuntimeError('A prepared worker exited before global stop')
            receipt['capture_finished'] = finished
            if stop(True)['stop_exit'] != 0:
                raise RuntimeError('External Nsight stop or closed-report check failed')
            record = wait_cell(240)
            receipt['launch_exit'] = launch.wait(timeout=60)
            if record['status'] != 'collected' or receipt['launch_exit'] != 0:
                raise RuntimeError('Cell or Nsight launch failed; preserve unqualified capture')
            receipt['exit_code'] = 0
        except BaseException as error:
            receipt['error'] = repr(error)
            if created and not stop_attempted:
                stop(False)
            identity = read_json(output / 'server-identity.json')
            if identity and not receipt.get('capture_finished') and same_live_process(identity):
                try:
                    os.kill(identity['pid'], signal.SIGTERM)
                    receipt['failure_cell_sigterm'] = identity
                except ProcessLookupError:
                    pass
            if identity:
                try:
                    wait_cell(240)
                except Exception as cleanup_error:
                    receipt['cleanup_error'] = repr(cleanup_error)
            if launch is not None:
                try:
                    receipt['launch_exit'] = launch.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    receipt['launch_exit'] = None
        finally:
            identity = read_json(output / 'server-identity.json')
            targets_gone = ((launch is None or launch.poll() is not None) and
                            (not identity or not same_live_process(identity)))
            if created and targets_gone:
                before = invoke('sessions-before-shutdown', plan['sessions'], 10)
                if before['returncode'] == 0 and session not in (output / 'sessions-before-shutdown.log').read_text():
                    receipt['session_shutdown'] = {'confirmed_absent': True, 'already_absent': True}
                else:
                    shutdown = invoke('shutdown', plan['shutdown'], 60)
                    after = invoke('sessions-after-shutdown', plan['sessions'], 10)
                    absent = (after['returncode'] == 0 and session not in (output / 'sessions-after-shutdown.log').read_text())
                    receipt['session_shutdown'] = {'returncode': shutdown['returncode'], 'confirmed_absent': absent}
                    if shutdown['returncode'] != 0 or not absent:
                        receipt['exit_code'] = 1
            elif created:
                receipt['session_shutdown'] = {'confirmed_absent': False,
                                               'error': 'Owned target exit is not confirmed; no shutdown issued'}
                receipt['exit_code'] = 1
            try:
                receipt['export'] = export_report(output, timeout=900)
            except Exception as error:
                receipt['export_error'] = repr(error)
                receipt['exit_code'] = 1
            receipt['finished_ns'] = time.time_ns()
            save(output / 'profile-receipt.json', receipt)
    return receipt
