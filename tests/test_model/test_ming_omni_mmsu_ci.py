# SPDX-License-Identifier: Apache-2.0
"""MMSU audio-in CI for Ming-Omni (Text + Audio -> Text, Talker OFF).

MMSU covers the Yuan voice-memo ASR requirement as a superset. PR #326 landed
audio-in support for Ming-Omni, but did not add a standalone ASR evaluation
path, so this stage uses the model-agnostic benchmark_omni_mmsu.py runner.

Ming requests text output here because this stage uses the Talker OFF launcher;
audio input is still supplied by the benchmark through the top-level ``audios``
field.

Usage:
    pytest tests/test_model/test_ming_omni_mmsu_ci.py -s -x
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from sglang_omni.utils import find_available_port
from tests.utils import (
    ServerHandle,
    apply_slack,
    assert_speed_thresholds,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "inclusionAI/Ming-flash-omni-2.0"
CONCURRENCY = 4
STARTUP_TIMEOUT = 2400
THINKER_TP_SIZE = 2
MMSU_MIN_ACCURACY = 0.40

# Thresholds carried over from PR #348; calibrate post-migration on H20.
_MMSU_P95 = {
    4: {
        "throughput_qps": 0.10,
        "tok_per_s_agg": 3.0,
        "latency_mean_s": 30.0,
    },
}
MMSU_THRESHOLDS = apply_slack(_MMSU_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Ming-Omni thinker-only server for audio-in understanding.

    This uses the Talker OFF launcher path because MMSU only needs text output.
    """
    port = find_available_port()
    log_file = tmp_path_factory.mktemp("server_logs") / "server.log"
    cmd = [
        sys.executable,
        "examples/run_ming_omni_server.py",
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--tp-size",
        str(THINKER_TP_SIZE),
        "--model-name",
        "ming-omni",
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


def _build_args(port: int, output_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=None,
        host="localhost",
        port=port,
        model="ming-omni",
        # This selects output modalities. The audio input is still supplied by
        # the benchmark via the top-level "audios" payload field.
        modalities="text",
        output_dir=output_dir,
        max_samples=50,
        task_names=None,
        categories=None,
        prompt=None,
        max_tokens=64,
        temperature=0.0,
        warmup=1,
        max_concurrency=CONCURRENCY,
        request_rate=float("inf"),
        timeout_s=300,
        save_audio=False,
        disable_tqdm=True,
        seed=None,
        repo_id=DATASETS["mmsu-ci-2000"],
        lang="en",
        asr_device="cuda:0",
    )


@pytest.mark.benchmark
def test_mmsu_accuracy_and_speed(
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run MMSU eval and assert accuracy and speed meet thresholds."""
    args = _build_args(server_process.port, str(tmp_path / "mmsu"))
    results = asyncio.run(run_mmsu(args))

    failed = results["accuracy"].get("failed_samples", 0)
    total = results["accuracy"].get("total_samples", 0)
    first_failures = [
        {
            "sample_id": sample.get("sample_id"),
            "latency_s": sample.get("latency_s"),
            "error": sample.get("error") or sample.get("raw_response"),
        }
        for sample in results.get("per_sample", [])
        if not sample.get("is_success")
    ][:3]
    assert failed == 0, (
        f"MMSU had {failed}/{total} failed requests (timeouts or empty responses); "
        f"any failure fails the test; first_failures={first_failures}"
    )

    accuracy = results["accuracy"]["overall_accuracy"]
    assert accuracy >= MMSU_MIN_ACCURACY, (
        f"MMSU accuracy {accuracy:.4f} ({accuracy * 100:.1f}%) < "
        f"threshold {MMSU_MIN_ACCURACY} ({MMSU_MIN_ACCURACY * 100:.0f}%)"
    )

    assert_speed_thresholds(results["speed"], MMSU_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
