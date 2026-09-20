"""Cyclic-input sustained protocol for the mps-evidence campaign (lane-parameterised).

Derived from round-1 sustained-v3 sustained_protocol.py (module name kept so the
unchanged client/analyzer/quality modules import it). Differences: campaign
paths, GPU UUID and ports come from the lane environment; arms cover Eager/Graph,
MPS on/off, DP1..DP4 and Green Context placements.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path

BASE = Path('/work/campaigns/mps-evidence-20260912-a01')
CODEX_BASE = Path('/work/campaigns/mps-host-gpu-green-context-20260912-a01')
DEPS = CODEX_BASE / 'deps'  # sglang 0.5.19 + flashinfer 0.6.18, read-only reuse
SOURCE = BASE / 'sources/evidence-9b148e05f'
SOURCE_SHA = '9b148e05fd19762fd6ef64fea5e98396a15be502'
RESULT_ROOT = Path('/work/results') / BASE.name / 'job-18012'
MODEL = '/shared-hf/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3'
ASR_MODEL = '/shared-hf/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5'
INPUT_SHA = 'e5c833a5d4885c77bb668eef4ad31049e1e8eb70262711c7782654e80dcac7e2'
SAMPLER_SHA_ORIGINAL = 'ef5edfefee0461b4849f10d55c5bd9ca2776995920ce70869f7c0a25ef04355e'  # Codex's sealed sampler
# Note (wenyao): sampler patched 2026-09-13T02:50Z to retry its MPS readiness probe (patches/sampler_probe_retry.py);
# preflight pins the patched file so an unnoticed edit still fails the run.
SAMPLER_SHA = '92622699c90ecf37675f4ca906b621671b80363c031c656b0ae0b4b33b8ec169'
FORBIDDEN_GPU = 'GPU-76fd1c0c-a95d-0947-c05f-985e5e9c7c0c'  # Codex session's GPU 0
FORBIDDEN_CPUS = set(range(0, 32))  # Codex session's replica/client/router cores

_GPU_RE = re.compile(r'GPU-[0-9a-fA-F-]{36}')


def lane_gpu_uuid() -> str:
    value = os.environ.get('LANE_GPU_UUID', '')
    if not _GPU_RE.fullmatch(value):
        raise RuntimeError('LANE_GPU_UUID must be one physical GPU UUID')
    if value == FORBIDDEN_GPU:
        raise RuntimeError('GPU 0 belongs to the other session')
    return value


GPU_UUID = os.environ.get('LANE_GPU_UUID', 'GPU-unset')
PORT_BASE = int(os.environ.get('LANE_PORT_BASE', '19000'))
ROUTER_PORT = PORT_BASE
REPLICA_PORT = PORT_BASE + 1  # replicas use REPLICA_PORT + rank
ASR_PORT = PORT_BASE + 10
ROUTER_URL = f'http://127.0.0.1:{ROUTER_PORT}'

# name -> (replicas, mps, graph, physical cores per replica)
ARMS = {
    'DP1-eager-off': (1, False, False, 8),
    'DP1-eager-off-16c': (1, False, False, 16),
    'DP1-graph-off': (1, False, True, 8),
    # Note (wenyao): single-replica controls with the MPS daemon on (added 2026-09-13T04:16Z): the cost of MPS itself.
    'DP1-eager-on': (1, True, False, 8),
    'DP1-graph-on': (1, True, True, 8),
    'DP2-eager-off': (2, False, False, 8),
    'DP2-eager-on': (2, True, False, 8),
    'DP2-graph-off': (2, False, True, 8),
    'DP2-graph-on': (2, True, True, 8),
    'DP3-graph-off': (3, False, True, 8),
    'DP3-graph-on': (3, True, True, 8),
    'DP4-graph-off': (4, False, True, 8),
    'DP4-graph-on': (4, True, True, 8),
}
PLACEMENTS = ('ordinary', 'indexed', 'union2')
SEND_SECONDS = 150
SETTLING_SECONDS = 30
BLOCK_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 300
ADMISSION_MESSAGES = ('The request queue is full.',
    'router overloaded: max in-flight requests reached', 'router upstream pool exhausted')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def label(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Label must use letters, digits, underscore, or hyphen')
    return value


def rate_label(rate):
    rate = float(rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Rate must be finite and positive')
    return format(rate, '.12g').replace('.', 'p')


def server_config(arm, placement=None, sms=None, rank=0, receipt_dir=None):
    replicas, mps, graph, _ = ARMS[arm]
    config = {'config_cls': 'Qwen3TTSPipelineConfig', 'name': 'qwen3-tts',
        'model_path': MODEL, 'mps': 'on' if mps else 'off', 'stages': {
        'tts_engine': {'engine': {'max_total_tokens': 32768, 'max_running_requests': 16,
            'max_queued_requests': 128, 'cuda_graph_max_bs': 16, 'disable_cuda_graph': not graph,
            'enable_torch_compile': False, 'mem_fraction_static': 0.85}},
        'vocoder': {'factory': {'initial_cuda_graph': graph, 'followup_cuda_graph': graph,
            'incremental_codec_cuda_graph': graph, 'incremental_codec_compile': False}}}}
    if placement is not None:
        if placement not in PLACEMENTS:
            raise ValueError('Unknown resource placement')
        if not graph:
            raise ValueError('Green Context placements require the Graph regime')
        if placement != 'ordinary' and (type(sms) is not int or sms <= 0):
            raise ValueError('indexed/union2 placements need a positive SM count')
        if receipt_dir is None:
            raise ValueError('receipt_dir required for resource placements')
        config['config_cls'] = 'Qwen3TTSEvidenceResourcePipelineConfig'
        config['stages']['tts_engine']['factory'] = dict(
            resource_placement=placement, resource_sms=None if placement == 'ordinary' else sms,
            resource_rank=rank, resource_group_count=replicas, resource_receipt_dir=str(receipt_dir))
    return config


def expected_actual_sms(placement, sms):
    if placement in (None, 'ordinary'):
        return 132
    if placement == 'indexed':
        return sms
    if placement == 'union2':
        return 2 * sms
    raise ValueError(placement)


def duration_offsets(rate, duration_s=SEND_SECONDS, seed=42):
    """Use the same independent NumPy RandomState Poisson process as round 1."""
    import numpy as np
    rate_label(rate)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('Duration must be finite and positive')
    rng, elapsed, offsets = np.random.RandomState(seed), 0., []
    while True:
        elapsed += float(rng.exponential(1.0 / float(rate)))
        ns = int(round(elapsed * 1e9))
        if ns >= round(duration_s * 1e9):
            return offsets
        offsets.append(ns)


def occurrences(samples, count, prefix, phase):
    """Never reuse a base sample ID as an output-file/ASR key."""
    label(prefix)
    if not samples or len({s.sample_id for s in samples}) != len(samples):
        raise ValueError('Base cohort must have unique IDs')
    rows, copies = [], []
    for i in range(count):
        base_index = 0 if phase == 'warmup' else i % len(samples)
        base = samples[base_index]
        occurrence = f'{prefix}-{phase}-{i:06d}'
        copies.append(replace(base, sample_id=occurrence))
        rows.append(dict(occurrence_id=occurrence, base_id=base.sample_id,
            base_index=base_index, cycle=i // len(samples), index=i, phase=phase,
            ref_audio=base.ref_audio, ref_text=base.ref_text, target_text=base.target_text))
    return copies, rows


def outcome(row):
    if row.get('result', {}).get('is_success'):
        return 'success'
    if row.get('exception_type') in ('TimeoutError', 'ServerTimeoutError', 'ConnectionTimeoutError', 'SocketTimeoutError'):
        return 'timeout'
    if row.get('exception_type') == 'CancelledError':
        return 'cancelled'
    if row.get('exception_type') == 'SendCutoff':
        return 'not_sent_before_cutoff'
    error = row.get('result', {}).get('error', '')
    if row.get('http_status') in (429, 503) and any(s in error for s in ADMISSION_MESSAGES):
        return 'admission_rejection'
    if row.get('http_status') is not None and row['http_status'] != 200:
        return 'http_error'
    if row.get('exception_type'):
        return 'client_exception'
    return 'generation_failure'


def lane_cpu_sets(replicas, cores_per_replica):
    """Physical cores from LANE_CPUS (e.g. '56-95'), disjoint from the other session."""
    spec = os.environ.get('LANE_CPUS', '')
    wanted = set()
    for part in spec.split(','):
        if not part:
            continue
        bounds = list(map(int, part.split('-')))
        wanted.update(range(bounds[0], bounds[-1] + 1))
    if not wanted:
        raise RuntimeError('LANE_CPUS must list the lane CPUs')
    if wanted & FORBIDDEN_CPUS:
        raise RuntimeError('LANE_CPUS overlaps the other session cores 0-31')
    allowed = os.sched_getaffinity(0)
    if not wanted <= allowed:
        raise RuntimeError(f'LANE_CPUS not all allowed: {sorted(wanted - allowed)}')
    physical = []
    for cpu in sorted(wanted):
        siblings = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text()
        first = int(re.split(r'[,-]', siblings.strip())[0])
        if first == cpu:
            physical.append(cpu)
    need = replicas * cores_per_replica + 8
    if len(physical) < need:
        raise RuntimeError(f'Lane has {len(physical)} physical cores, needs {need}')
    sets = [physical[i * cores_per_replica:(i + 1) * cores_per_replica] for i in range(replicas)]
    used = replicas * cores_per_replica
    return sets, physical[used:used + 4], physical[used + 4:used + 8]


# Nsight external-stop pairing: the cell must outwait sessions(10)+status(10)+stop(NSYS_STOP_TIMEOUT_S).
NSYS_STOP_TIMEOUT_S = 300
NSYS_STOP_ACK_WAIT_S = NSYS_STOP_TIMEOUT_S + 20 + 80
# Node-wide boot serialisation (no PID namespace; /tmp shared across container instances).
RAM_BOOT_LOCK = '/tmp/evid-ram-boot.lock'
LANE_LOCK_DIR = '/tmp'


def cgroup_anon_gb():
    """Anonymous (non-reclaimable) memory charged to the job cgroup, or None outside cgroup v2."""
    try:
        for line in open('/sys/fs/cgroup/memory.stat'):
            key, value = line.split()
            if key == 'anon':
                return int(value) / 1e9
    except (OSError, ValueError):
        return None
    return None


def cgroup_limit_gb():
    try:
        text = open('/sys/fs/cgroup/memory.max').read().strip()
        return None if text == 'max' else int(text) / 1e9
    except (OSError, ValueError):
        return None


def wait_ports_free(ports, timeout_s=240, log=print):
    """Block until every lane port passes the launcher's plain bind() test.

    sglang_omni's launcher probes the requested port without SO_REUSEADDR and silently
    falls back to a random port when the probe fails; TIME_WAIT sockets left by the
    previous run's router->replica connections make that probe fail for ~60 s.
    """
    import socket
    import time
    deadline = time.monotonic() + timeout_s
    while True:
        busy = []
        for port in ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(('127.0.0.1', port))
                except OSError:
                    busy.append(port)
        if not busy:
            return
        log(json.dumps({'port_wait': True, 'busy_ports': busy}), flush=True)
        if time.monotonic() > deadline:
            raise RuntimeError(f'Lane ports still busy after {timeout_s}s: {busy}')
        time.sleep(5)


def host_rss_gb():
    total = 0
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                total += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
            except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
                continue
    return total / 2**30


def raise_nofile_limit():
    """Popen preexec_fn: lift the soft open-files limit to the hard limit.

    The router relay needs more than 2 x upstream_pool_size (512) descriptors under load;
    the inherited soft limit of 1024 produced bursts of ServerDisconnectedError / http_error
    outcomes at high offered rates (DP2-graph-off sweep1 r32, DP2-graph-on r1 r40).
    """
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
