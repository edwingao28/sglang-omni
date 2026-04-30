# SPDX-License-Identifier: Apache-2.0
"""TTS CI for Ming-Omni: RTF performance + WER accuracy in a single session.

This borrows the Qwen3-Omni single-job pattern: a module-scoped server
fixture, the perf test runs first via ``speed_output_dir``, and the WER test
reuses the generated audio from the same output directory after stopping the
server to free GPU memory for ASR transcription.

Usage:
    pytest tests/test_model/test_ming_omni_tts_ci.py -s -x -v
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    run_omni_seedtts_benchmark,
)
from sglang_omni.utils import find_available_port
from tests.utils import (
    apply_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_summary_metrics,
    assert_wer_results,
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "inclusionAI/Ming-flash-omni-2.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONCURRENCY = 1
MAX_SAMPLES = 10
DATASET_CACHE_ENV = "SGLANG_SEEDTTS_MINI_DIR"
STARTUP_TIMEOUT = 2400
WER_TIMEOUT = 600
THINKER_TP_SIZE = 1
TALKER_GPU_ID = 1
THINKER_CPU_OFFLOAD_GB_ENV = "MING_TTS_CPU_OFFLOAD_GB"
THINKER_MEM_FRACTION_STATIC_ENV = "MING_TTS_MEM_FRACTION_STATIC"

# TODO (wenyao): Placeholder P95 — replace with measured values in follow-up
# reference-run PR. Anchored to RTF≈0.06.
# (wenyao) rtf_mean = wall_time / audio_duration (higher worse); slack 1.25 = fail-bound multiplier.
_VC_NON_STREAM_P95 = {
    1: {
        "throughput_qps": 0.15,
        "tok_per_s_agg": 2.0,
        "latency_mean_s": 8.0,
        "rtf_mean": 0.06,
    },
}

THRESHOLD_SLACK_HIGHER = 0.75
THRESHOLD_SLACK_LOWER = 1.25
VC_NON_STREAM_THRESHOLDS = apply_slack(
    _VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
NONCLONE_WER_MAX_CORPUS = 0.20
NONCLONE_WER_MAX_PER_SAMPLE = 0.60


def _assert_ttfa_diagnostics(summary: dict, per_request: list[dict]) -> None:
    """Verify non-stream TTFA baseline metrics are emitted.

    Ming does not support true streaming TTS yet, so these values are diagnostics
    rather than a blocking performance gate.
    """
    for key in (
        "ttfa_mean_s",
        "ttfa_median_s",
        "ttfa_p95_s",
        "ttfa_p99_s",
    ):
        assert summary.get(key, 0) > 0, f"Expected positive {key}, got {summary}"

    for req in per_request:
        rid = req["id"]
        ttfa = req.get("ttfa_s")
        assert (
            ttfa is not None and ttfa > 0
        ), f"Request {rid}: ttfa_s={ttfa}, expected > 0"


def _run_benchmark(
    port: int,
    meta: str,
    output_dir: str,
) -> dict:
    config = OmniSeedttsBenchmarkConfig(
        model="ming-omni",
        port=port,
        meta=meta,
        output_dir=output_dir,
        max_samples=MAX_SAMPLES,
        # (wenyao) Ming has no per-request voice-clone codepath.
        voice_clone=False,
    )
    speed_results = asyncio.run(run_omni_seedtts_benchmark(config))
    assert (
        "summary" in speed_results
    ), f"Missing 'summary' key in results. Keys: {list(speed_results.keys())}"
    assert (
        "per_request" in speed_results
    ), f"Missing 'per_request' key in results. Keys: {list(speed_results.keys())}"
    return speed_results


def _run_wer_transcribe(
    meta_path: str,
    output_dir: str,
    lang: str = "en",
    device: str = "cuda:0",
) -> dict:
    """Transcribe saved audio and compute WER in CI."""
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
        "--transcribe-only",
        "--meta",
        meta_path,
        "--output-dir",
        output_dir,
        "--model",
        "ming-omni",
        "--lang",
        lang,
        "--device",
        device,
    ]

    env = no_proxy_env()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT)
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=WER_TIMEOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"WER transcribe failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    results_path = Path(output_dir) / "wer_results.json"
    assert results_path.exists(), f"WER results file not found: {results_path}"

    with open(results_path) as f:
        wer_results = json.load(f)
    assert (
        "summary" in wer_results
    ), f"Missing 'summary' key in WER results. Keys: {list(wer_results.keys())}"
    assert (
        "per_sample" in wer_results
    ), f"Missing 'per_sample' key in WER results. Keys: {list(wer_results.keys())}"

    summary = wer_results["summary"]
    if summary.get("skipped", 0) > 0:
        print(
            f"\n[WER DIAGNOSTIC] {summary['skipped']}/{summary['total_samples']} "
            f"samples skipped.\nSubprocess stderr:\n{result.stderr}"
        )
        for sample in wer_results["per_sample"]:
            if not sample.get("is_success", True):
                print(f"  FAILED sample {sample['id']}: {sample.get('error')}")

    return wer_results


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    override_dir = os.environ.get(DATASET_CACHE_ENV)
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        root = tmp_path_factory.mktemp("seed_tts_eval") / "data"
    download_dataset(DATASETS["seedtts-mini"], str(root), quiet=True)
    return root


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Ming-Omni speech server and wait until healthy."""
    port = find_available_port()
    log_file = tmp_path_factory.mktemp("server_logs") / "server.log"
    cmd = [
        sys.executable,
        "examples/run_ming_omni_speech_server.py",
        "--model-path",
        MODEL_PATH,
        "--gpu-thinker",
        "0",
        "--gpu-talker",
        str(TALKER_GPU_ID),
        "--tp-size",
        str(THINKER_TP_SIZE),
        "--port",
        str(port),
        "--model-name",
        "ming-omni",
    ]
    cpu_offload_gb = os.environ.get(THINKER_CPU_OFFLOAD_GB_ENV)
    if cpu_offload_gb:
        cmd.extend(["--cpu-offload-gb", cpu_offload_gb])
    mem_fraction_static = os.environ.get(THINKER_MEM_FRACTION_STATIC_ENV)
    if mem_fraction_static:
        cmd.extend(["--mem-fraction-static", mem_fraction_static])
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def speed_output_dir(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """Run the speed benchmark once and expose its output directory."""
    output_dir = str(tmp_path_factory.mktemp("nonclone_nonstream"))
    results = _run_benchmark(
        server_process.port,
        str(dataset_dir / "en" / "meta.lst"),
        output_dir,
    )
    summary, per_request = results["summary"], results["per_request"]
    assert_summary_metrics(summary)
    assert_per_request_fields(per_request)
    _assert_ttfa_diagnostics(summary, per_request)
    assert_speed_thresholds(summary, VC_NON_STREAM_THRESHOLDS, CONCURRENCY)
    return output_dir


@pytest.fixture(scope="module")
def wer_audio_dir(
    server_process: subprocess.Popen,
    speed_output_dir: str,
) -> str:
    """Reuse perf audio for WER after freeing the Ming server GPUs.

    This keeps perf and WER in one Ming server startup while avoiding GPU
    contention between the live thinker on cuda:0 and ASR transcription.
    ``stop_server`` is idempotent, so the normal server fixture teardown can
    still call it.
    """
    stop_server(server_process)
    generated_path = Path(speed_output_dir) / "generated.json"
    assert generated_path.exists(), f"WER metadata missing: {generated_path}"
    return speed_output_dir


@pytest.mark.benchmark
def test_tts_non_streaming_perf(speed_output_dir: str) -> None:
    """Smoke check: the speed-benchmark fixture asserts metrics/thresholds."""
    assert Path(speed_output_dir).is_dir()


@pytest.mark.benchmark
def test_tts_wer(
    wer_audio_dir: str,
    dataset_dir: Path,
) -> None:
    results = _run_wer_transcribe(
        str(dataset_dir / "en" / "meta.lst"),
        wer_audio_dir,
    )
    assert_wer_results(results, NONCLONE_WER_MAX_CORPUS, NONCLONE_WER_MAX_PER_SAMPLE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
