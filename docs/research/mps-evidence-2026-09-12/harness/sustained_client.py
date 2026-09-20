"""Cyclic sustained traffic using the unchanged upstream TTS send function.

mps-evidence copy of round-1 sustained_client.py; only the router URL comes from the lane protocol.

Only the resident controller starts services. This client never allocates GPUs.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time

from sustained_protocol import (INPUT_SHA, REQUEST_TIMEOUT_SECONDS, ROUTER_URL, SEND_SECONDS,
    SETTLING_SECONDS, digest, duration_offsets, label, occurrences, outcome, sha, write_new)


def stamp(row, name):
    row[name + '_ns'] = time.time_ns()
    row[name + '_mono_ns'] = time.perf_counter_ns()


class ObservedContext:
    """Observe aiohttp without changing payloads, retries, response reads or UUIDs."""
    def __init__(self, context, row):
        self.context, self.row = context, row

    async def __aenter__(self):
        try:
            response = await self.context.__aenter__()
        except BaseException as exc:
            self.row['exception_type'] = type(exc).__name__
            raise
        stamp(self.row, 'headers')
        self.row['http_status'] = response.status
        return response

    async def __aexit__(self, kind, exc, tb):
        if exc is not None:
            self.row['exception_type'] = type(exc).__name__
        try:
            return await self.context.__aexit__(kind, exc, tb)
        finally:
            stamp(self.row, 'http_context_end')


class ObservedSession:
    def __init__(self, session, row):
        self.session, self.row = session, row

    def post(self, *args, **kwargs):
        if time.perf_counter_ns() >= self.row.get('send_cutoff_mono_ns', float('inf')):
            raise SendCutoff('Payload preparation passed the fixed send cutoff')
        stamp(self.row, 'send')
        self.row['client_request_uuid'] = kwargs.get('headers', {}).get('X-Request-ID')
        return ObservedContext(self.session.post(*args, **kwargs), self.row)


class SendCutoff(RuntimeError):
    pass


DISPATCH_GRACE_S = 2.0  # event-loop lag reaches 1.6 s at DP4 r40 (6k in-flight completions); 0.5 s dropped the last arrivals


async def send_one(session, sample, send_fn, row, journal):
    stamp(row, 'task_start')
    try:
        row['result'] = asdict(await send_fn(ObservedSession(session, row), sample))
    except BaseException as exc:
        row.update(exception_type=type(exc).__name__, exception=repr(exc))
        # Cancellation must be recorded too; the parent scheduler owns cancellation.
    finally:
        stamp(row, 'terminal')
        row['outcome'] = outcome(row)
        journal.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row


async def dispatch(session, samples, send_fn, offsets, duration_s, journal,
                   drain_s=REQUEST_TIMEOUT_SECONDS):
    """Absolute open-loop arrivals, hard send cutoff, bounded drain, complete ledger."""
    origin = {'wall_ns': time.time_ns(), 'mono_ns': time.perf_counter_ns()}
    cutoff = origin['mono_ns'] + round(duration_s * 1e9)
    # Arrivals are scheduled strictly before the cutoff; a dispatch tick that wakes a few ms late must
    # still send them (otherwise the last dense arrivals of a high-rate cell surface as not_dispatched).
    # Sends after the nominal cutoff fall outside the [30,150) window by their actual send time.
    grace_ns = round(DISPATCH_GRACE_S * 1e9)
    rows = [dict(index=i, occurrence_id=s.sample_id, nominal_offset_ns=offset, send_cutoff_mono_ns=cutoff + grace_ns,
                 outcome='not_dispatched') for i, (s, offset) in enumerate(zip(samples, offsets))]
    tasks = []
    status, error = 'collected', None
    progress = None

    async def report_progress():
        for second in range(30, int(duration_s) + 1, 30):
            await asyncio.sleep(max(0, (origin['mono_ns'] + second * 10**9 - time.perf_counter_ns()) / 1e9))
            journal.flush()
            print(json.dumps({'elapsed_s': second, 'scheduled': len(rows),
                'sent': sum('send_ns' in r for r in rows),
                'terminal': sum('terminal_ns' in r for r in rows),
                'success': sum(r['outcome'] == 'success' for r in rows)}), flush=True)

    try:
        progress = asyncio.create_task(report_progress())
        for sample, row in zip(samples, rows):
            deadline = origin['mono_ns'] + row['nominal_offset_ns']
            await asyncio.sleep(max(0, (deadline - time.perf_counter_ns()) / 1e9))
            if time.perf_counter_ns() >= cutoff + grace_ns:
                break
            stamp(row, 'dispatch')
            row['dispatch_lag_ns'] = row['dispatch_mono_ns'] - deadline
            tasks.append(asyncio.create_task(send_one(session, sample, send_fn, row, journal)))
        await asyncio.sleep(max(0, (cutoff - time.perf_counter_ns()) / 1e9))
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=max(0, (cutoff + round(drain_s * 1e9) - time.perf_counter_ns()) / 1e9))
            if pending:
                status = 'drain_deadline'
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
    except BaseException as exc:
        status, error = 'interrupted', repr(exc)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if progress:
            progress.cancel()
            await asyncio.gather(progress, return_exceptions=True)
        for row in rows:
            if 'terminal_ns' not in row:
                row['outcome'] = 'not_dispatched'
                journal.write(json.dumps(row) + '\n')
        journal.flush()
    return dict(status=status, error=error, origin=origin, duration_s=duration_s, dispatch_grace_s=DISPATCH_GRACE_S,
        drain_timeout_s=drain_s, finished_ns=time.time_ns(), finished_mono_ns=time.perf_counter_ns(),
        per_request=rows)


async def prepare(session, samples, send_fn, journal):
    rows = [dict(index=i, occurrence_id=s.sample_id, outcome='not_dispatched') for i, s in enumerate(samples)]
    semaphore = asyncio.Semaphore(16)

    async def bounded(sample, row):
        async with semaphore:
            return await send_one(session, sample, send_fn, row, journal)

    status, error = 'prepared', None
    try:
        # Preserve upstream's three concurrent first-input warmups; unique file IDs only.
        await asyncio.gather(*(send_one(session, s, send_fn, row, journal) for s, row in zip(samples[:3], rows[:3])))
        if any(r['outcome'] != 'success' for r in rows[:3]):
            status = 'warmup_failed'
        else:
            for start in (3, 131, 259):
                await asyncio.gather(*(bounded(s, r) for s, r in zip(samples[start:start+128], rows[start:start+128])))
                journal.flush()
                if any(r['outcome'] != 'success' for r in rows[start:start+128]):
                    status = 'conditioning_failed'
                    break
    except BaseException as exc:
        status, error = 'interrupted', repr(exc)
    finally:
        for row in rows:
            if 'terminal_ns' not in row:
                journal.write(json.dumps(row) + '\n')
        journal.flush()
    return dict(status=status, error=error, per_request=rows)


async def run(args):
    import aiohttp
    from benchmarks.benchmarker.data import RequestResult
    from benchmarks.eval.benchmark_tts_seedtts import (
        TtsSeedttsBenchmarkConfig, _build_generation_kwargs, _build_results_config, _load_benchmark_samples)
    from benchmarks.metrics.performance import compute_speed_metrics
    from benchmarks.tasks.tts import make_tts_send_fn, save_generated_audio_metadata, save_speed_results

    if sha(args.meta) != INPUT_SHA:
        raise ValueError('Sealed base meta SHA mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    config = TtsSeedttsBenchmarkConfig(model='qwen3-tts', meta=str(args.meta),
        base_url=ROUTER_URL, max_samples=128, concurrency=0,
        warmup=3, seed=42, max_new_tokens=2048, request_rate=args.rate or float('inf'),
        disable_tqdm=True, output_dir=str(args.output))
    base = _load_benchmark_samples(config)
    if len(base) != 128:
        raise ValueError('Exactly 128 sealed base inputs required')
    offsets = None
    if args.prepare:
        warmup, warmup_manifest = occurrences(base, 3, args.prefix, 'warmup')
        conditioning, condition_manifest = occurrences(base, 384, args.prefix, 'conditioning')
        samples, occurrence_rows = warmup + conditioning, warmup_manifest + condition_manifest
    else:
        offsets = duration_offsets(args.rate)
        samples, occurrence_rows = occurrences(base, len(offsets), args.prefix, 'service')
    manifest = dict(schema_version=1, prefix=args.prefix, input_sha256=INPUT_SHA,
        base_sample_ids=[s.sample_id for s in base], occurrences=occurrence_rows,
        phase='prepare' if args.prepare else 'service', seed=42, requested_rate=args.rate,
        duration_s=None if args.prepare else SEND_SECONDS, offsets_ns=offsets)
    write_new(args.output / 'occurrence-manifest.json', manifest)
    config_record = _build_results_config(config, base_url=config.base_url)
    config_record.update(request_rate=args.rate, warmup=3 if args.prepare else 0,
        max_samples=len(samples), preparation='resident 3 warmup + 3x128 conditioning before service cells')
    write_new(args.output / 'client-config.json', config_record)
    audio = args.output / 'audio'
    audio.mkdir()
    send_fn = make_tts_send_fn(config.model, config.base_url + '/v1/audio/speech',
        response_format=config.response_format, stream=False, initial_codec_chunk_frames=None,
        no_ref_audio=False, ref_format=config.ref_format, no_ref_text=False,
        voice=None, task_type=None, instructions=None, save_audio_dir=str(audio.resolve()),
        **_build_generation_kwargs(config))
    capture = None
    start = time.perf_counter()
    current = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, current.cancel)
    with (args.output / 'terminal-journal.jsonl').open('x', buffering=65536) as journal:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
                                         connector=aiohttp.TCPConnector(limit=0)) as session:
            capture = await prepare(session, samples, send_fn, journal) if args.prepare else await dispatch(
                session, samples, send_fn, offsets, SEND_SECONDS, journal)
    capture.update(schema_version=1, manifest_sha256=sha(args.output / 'occurrence-manifest.json'),
        schedule_sha256=digest({'occurrence_ids': [s.sample_id for s in samples], 'offsets_ns': offsets}),
        settling_s=SETTLING_SECONDS, concurrency=0, seed=42, requested_rate=args.rate,
        latency_semantics='send is session.post invocation; terminal follows upstream WAV parsing/write. Upstream latency and wall timestamps retained separately.')
    write_new(args.output / 'sustained-capture.json', capture)
    results = []
    for sample, row in zip(samples, capture['per_request']):
        fields = row.get('result') or dict(request_id=sample.sample_id,
            text=sample.target_text[:60], error=row.get('exception') or row['outcome'])
        results.append(RequestResult(**fields))
    save_generated_audio_metadata(results, samples, str(args.output))
    # Historical full-cohort summary is retained as diagnostic, never as fixed-window throughput.
    save_speed_results(results, compute_speed_metrics(results, wall_clock_s=time.perf_counter()-start),
                       config_record, str(args.output))
    print(json.dumps({'status': capture['status'], 'planned': len(samples),
                      'outcomes': {key: sum(r['outcome'] == key for r in capture['per_request'])
                                   for key in sorted({r['outcome'] for r in capture['per_request']})}}), flush=True)
    return 0 if capture['status'] in ('collected', 'prepared') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--prefix', type=label, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--rate', type=float)
    args = parser.parse_args()
    if args.prepare == (args.rate is not None):
        parser.error('Select exactly one of --prepare or --rate')
    if args.rate is not None:
        duration_offsets(args.rate, .001)
    raise SystemExit(asyncio.run(run(args)))


if __name__ == '__main__':
    main()
