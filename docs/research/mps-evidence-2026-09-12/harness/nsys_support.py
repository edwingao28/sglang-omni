"""One campaign's Nsight launch and closed-report export; no environment changes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

NSYS = '/opt/nvidia/nsight-systems-cli/2026.4.1/bin/nsys'


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'pgid': int(fields[2]), 'start_ticks': int(fields[19])}


def wrap_argv(argv, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError('Nsight output must be fresh: ' + str(output))
    flags = ['--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
             '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi',
             '--capture-range-end=stop', '--cuda-event-trace=false',
             '--cuda-memory-usage=false', '--trace-fork-before-exec=false',
             '--flush-on-cudaprofilerstop=false', '--discard-environment=true',
             '--wait=all', '--kill=none', '--force-overwrite=false',
             '--output=' + str(output/'capture')]
    command = [NSYS, 'profile', *flags, sys.executable, '-S', '-B',
               str(Path(__file__).with_name('server_entry.py')),
               str(output/'server-identity.json'), *map(str, argv)]
    save(output/'command.json', {'argv': command, 'created_ns': time.time_ns(),
         'scope': 'CUDA graph node kernels, eager kernels, CUDA API calls and request NVTX; not SM utilization counters'})
    return command


def export_report(output, timeout=90):
    output = Path(output)
    report = output/'capture.nsys-rep'
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError('Nsight report is missing after profiler exit')
    database = output/'capture.sqlite'
    argv = [NSYS, 'export', '--type=sqlite', '--force-overwrite=false',
            '--output=' + str(database), str(report)]
    with (output/'export.log').open('xb') as log:
        proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode:
        raise RuntimeError('Nsight SQLite export failed: ' + str(proc.returncode))
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {name: db.execute('SELECT count(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
                  for name in tables if name.startswith(('CUPTI_', 'NVTX_', 'PROCESSES', 'THREADS'))}
    files = []
    for path in (report, database):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
        files.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    receipt = {'closed': True, 'export_exit': proc.returncode, 'files': files,
               'tables': tables, 'activity_counts': counts, 'finished_ns': time.time_ns()}
    save(output/'export-receipt.json', receipt)
    return receipt


def live_group_members(pgid):
    """Read only the recorded target group; never signal an unverified member."""
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            item = process_identity(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if item['pgid'] == pgid and item['state'] != 'Z':
            members.append(item)
    return sorted(members, key=lambda item: item['pid'])


def finish_server(owned_command, output, timeout=90, *, capture_requested=None):
    """Close Nsight and drain its real serve group.

    Only a caller that has not started its profile-capable client may pass False.
    None/True requires a report; absence of a start marker alone proves nothing.
    """
    output = Path(output)
    prior = output/'finish-receipt.json'
    if prior.exists():
        receipt = json.loads(prior.read_text())
        if receipt.get('closed'):
            return receipt
        raise RuntimeError('prior Nsight finalization failed; preserve attempt')
    receipt = {'started_ns': time.time_ns(), 'supervisor_pid': owned_command.process.pid,
               'closed': False, 'signal_sent': False,
               'capture_requested': capture_requested}
    try:
        identity = json.loads((output/'server-identity.json').read_text())
        receipt['server'] = identity
        receipt['real_group_before'] = live_group_members(identity['pgid'])
        try:
            current = process_identity(identity['pid'])
        except (FileNotFoundError, ProcessLookupError):
            current = None
        if current and current['state'] != 'Z':
            if (current['start_ticks'] != identity['start_ticks'] or
                    current['pgid'] != identity['pgid']):
                raise RuntimeError('recorded serve identity changed; refusing signal')
            os.kill(identity['pid'], signal.SIGTERM)
            receipt['signal_sent'] = True
        deadline = time.monotonic() + timeout
        while owned_command.returncode() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('serve/Nsight did not exit before report-close deadline')
            time.sleep(.2)
        receipt['profiler_exit'] = owned_command.returncode()
        # Some launchers retain the wrapper's group; its held supervisor is ours.
        # With Nsight's separate serve group, do not substitute wrapper drain proof.
        if identity['pgid'] == owned_command.process.pid:
            owned_command.stop()
        while True:
            remaining = live_group_members(identity['pgid'])
            receipt['real_group_remaining'] = remaining
            if not remaining:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('recorded real serve process group did not drain')
            time.sleep(.2)
        receipt['real_group_drained'] = True
        payload = output.parent
        evidence = list((payload/'profile-markers').glob('profile-*-started.json'))
        evidence += [payload/name for name in ('start_profile.json', 'profile-control.json')
                     if (payload/name).exists()]
        receipt['capture_evidence'] = [str(path) for path in sorted(evidence)]
        report = output/'capture.nsys-rep'
        if (not report.exists() and capture_requested is False and not evidence and
                receipt['profiler_exit'] == 0):
            receipt.update(closed=True, capture_status='not_requested', report_closed=False,
                           export_exit=None, files=[], tables=[], activity_counts={},
                           finished_ns=time.time_ns())
            save(output/'uncaptured-receipt.json', receipt)
        else:
            # A SIGTERM target exit may propagate through nsys; actual report closure
            # remains mandatory after capture was requested or when that is unknown.
            receipt.update(export_report(output, timeout=timeout))
            receipt.update(capture_status='captured', report_closed=True)
        save(prior, receipt)
        return receipt
    except BaseException as exc:
        if 'server' in receipt:
            receipt['real_group_remaining'] = live_group_members(receipt['server']['pgid'])
        receipt['error'] = repr(exc)
        save(prior, receipt)
        raise
