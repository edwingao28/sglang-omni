"""Apply the MPS probe-tolerance patch to sglang_omni/pipeline/mp_runner.py (idempotent).

  python3 mps_probe_tolerance.py <path-to-mp_runner.py>
A single failed `get_server_list` control query (daemon and server alive) killed two
replicas on 2026-09-13 01:12:01Z; the monitor now requires three consecutive failed
probes (15 s) before failing the pipeline.
"""
import sys
from pathlib import Path

OLD_HEAD = '''logger = logging.getLogger(__name__)
'''
NEW_HEAD = '''logger = logging.getLogger(__name__)

# Note (wenyao): one failed `get_server_list` query has been observed while the MPS
# control daemon and server were alive and every client kept running; fail only after
# this many consecutive probe failures (5 s apart) instead of on the first one.
_MPS_PROBE_FAILURE_TOLERANCE = 3
_MONITOR_INTERVAL_S = 5.0
'''
OLD_LOOP = '''    async def _monitor_children(self) -> None:
        while self._started:
'''
NEW_LOOP = '''    async def _monitor_children(self) -> None:
        consecutive_probe_failures = 0
        while self._started:
'''
OLD_PROBE = '''            if self._mps is not None:
                probe_failures = await self._mps.probe_failures()
                if probe_failures:
                    details = "; ".join(
                        f"{gpu_uuid}: {reason}"
                        for gpu_uuid, reason in sorted(probe_failures.items())
                    )
                    error = RuntimeError(
                        f"MPS health check failed on physical GPU(s) ({details}); "
                        "failing the pipeline instead of serving degraded"
                    )
                    logger.error("%s", error)
                    await self._fail_runtime(error)
                    return
            await asyncio.sleep(5.0)
'''
NEW_PROBE = '''            if self._mps is not None:
                probe_failures = await self._mps.probe_failures()
                if probe_failures:
                    consecutive_probe_failures += 1
                    details = "; ".join(
                        f"{gpu_uuid}: {reason}"
                        for gpu_uuid, reason in sorted(probe_failures.items())
                    )
                    if consecutive_probe_failures < _MPS_PROBE_FAILURE_TOLERANCE:
                        logger.warning(
                            "MPS health probe failed (%d/%d consecutive) on physical "
                            "GPU(s) (%s); probing again before failing the pipeline",
                            consecutive_probe_failures,
                            _MPS_PROBE_FAILURE_TOLERANCE,
                            details,
                        )
                    else:
                        error = RuntimeError(
                            f"MPS health check failed on physical GPU(s) ({details}) "
                            f"{consecutive_probe_failures} consecutive times; "
                            "failing the pipeline instead of serving degraded"
                        )
                        logger.error("%s", error)
                        await self._fail_runtime(error)
                        return
                else:
                    consecutive_probe_failures = 0
            await asyncio.sleep(_MONITOR_INTERVAL_S)
'''


def main():
    path = Path(sys.argv[1])
    text = path.read_text()
    if '_MONITOR_INTERVAL_S' in text:
        print(f'{path}: already patched')
        return
    if '_MPS_PROBE_FAILURE_TOLERANCE' in text:  # v1 applied earlier: only add the interval constant
        hunks = (('_MPS_PROBE_FAILURE_TOLERANCE = 3\n', '_MPS_PROBE_FAILURE_TOLERANCE = 3\n_MONITOR_INTERVAL_S = 5.0\n'),
                 ('                    consecutive_probe_failures = 0\n            await asyncio.sleep(5.0)\n',
                  '                    consecutive_probe_failures = 0\n            await asyncio.sleep(_MONITOR_INTERVAL_S)\n'))
    else:
        hunks = ((OLD_HEAD, NEW_HEAD), (OLD_LOOP, NEW_LOOP), (OLD_PROBE, NEW_PROBE))
    for old, new in hunks:
        if text.count(old) != 1:
            raise SystemExit(f'{path}: expected exactly one match for a patch hunk, found {text.count(old)}')
        text = text.replace(old, new)
    path.write_text(text)
    print(f'{path}: patched')


if __name__ == '__main__':
    main()
