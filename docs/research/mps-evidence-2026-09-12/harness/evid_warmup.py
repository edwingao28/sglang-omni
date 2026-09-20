"""Run upstream's three first-sample warmups through the lane router; keep evidence.

Copy of round-1 profile-warmup-v2.py with the router URL from the lane protocol.
"""
import asyncio
import json
import sys
from pathlib import Path

import aiohttp

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.tts import make_tts_send_fn
from sustained_protocol import ROUTER_URL


async def main():
    output_path = Path(sys.argv[2])
    if output_path.exists():
        raise FileExistsError(output_path)
    sample = load_seedtts_samples(sys.argv[1], 1)[0]
    send = make_tts_send_fn('qwen3-tts', ROUTER_URL + '/v1/audio/speech', max_new_tokens=2048, seed=42)
    runner = BenchmarkRunner(RunConfig(max_concurrency=16, warmup=3, disable_tqdm=True))
    evidence = []

    async def record_send(session, sample):
        row = {'id': None, 'ok': False, 'error': 'incomplete', 'latency_s': None}
        evidence.append(row)
        try:
            result = await send(session, sample)
            row.update(id=result.request_id, ok=result.is_success, error=result.error, latency_s=result.latency_s)
            return result
        except BaseException as exc:
            row['error'] = repr(exc)
            raise

    try:
        timeout = aiohttp.ClientTimeout(total=runner.config.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await runner._warmup(session, [sample], record_send)
        assert len(evidence) == 3 and all(row['ok'] for row in evidence), evidence
    finally:
        with output_path.open('x') as output:
            json.dump(evidence, output, indent=2)
            output.write('\n')


if __name__ == '__main__':
    asyncio.run(main())
