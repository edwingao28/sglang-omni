# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark TTFA diagnostics."""

from __future__ import annotations

import asyncio
import base64
import importlib.machinery
import importlib.util
import io
import json
import sys
import types
import wave

import pytest


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


_scipy_signal = _stub_module("scipy.signal")
_inserted_module_stubs: list[str] = []


def _install_missing_module_stub(name: str, module) -> None:
    if name in sys.modules:
        return
    try:
        available = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        available = False
    if not available:
        sys.modules[name] = module
        _inserted_module_stubs.append(name)


for _module_name, _module in {
    "aiohttp": _stub_module("aiohttp", ClientError=Exception, ClientSession=object),
    "jiwer": _stub_module("jiwer", process_words=lambda *args, **kwargs: None),
    "requests": _stub_module(
        "requests",
        get=lambda *args, **kwargs: None,
        exceptions=types.SimpleNamespace(RequestException=Exception),
    ),
    "scipy": _stub_module("scipy", signal=_scipy_signal),
    "scipy.signal": _scipy_signal,
    "soundfile": _stub_module(
        "soundfile", info=lambda path: types.SimpleNamespace(duration=0.1)
    ),
}.items():
    _install_missing_module_stub(_module_name, _module)

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_omni_seedtts
from benchmarks.eval.benchmark_omni_seedtts import make_send_fn
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks import tts as tts_tasks
from benchmarks.tasks.tts import (
    _handle_streaming_response,
    build_speed_results,
    save_speed_results,
)

for _module_name in _inserted_module_stubs:
    sys.modules.pop(_module_name, None)


def _wav_bytes(*, sample_rate: int = 16000, frames: int = 1600) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def test_compute_speed_metrics_summarizes_ttfa() -> None:
    outputs = [
        RequestResult(
            request_id="a",
            is_success=True,
            latency_s=2.0,
            audio_duration_s=1.0,
            rtf=2.0,
            ttfa_s=0.4,
        ),
        RequestResult(
            request_id="b",
            is_success=True,
            latency_s=4.0,
            audio_duration_s=2.0,
            rtf=2.0,
            ttfa_s=0.8,
        ),
    ]

    summary = compute_speed_metrics(outputs, wall_clock_s=6.0)

    assert summary["ttfa_mean_s"] == 0.6
    assert summary["ttfa_median_s"] == 0.6
    assert summary["ttfa_p95_s"] == 0.78
    assert summary["ttfa_p99_s"] == 0.796


def test_speed_results_serialize_ttfa(tmp_path) -> None:
    output = RequestResult(
        request_id="sample-1",
        text="hello",
        is_success=True,
        latency_s=1.0,
        audio_duration_s=0.5,
        rtf=2.0,
        ttfa_s=0.34567,
        wav_path="/tmp/sample-1.wav",
    )
    metrics = compute_speed_metrics([output], wall_clock_s=1.0)

    results = build_speed_results([output], metrics, {"model": "unit-test"})
    assert results["per_request"][0]["ttfa_s"] == 0.3457

    save_speed_results([output], metrics, {"model": "unit-test"}, str(tmp_path))

    speed_json = json.loads((tmp_path / "speed_results.json").read_text())
    assert speed_json["summary"]["ttfa_mean_s"] == 0.346
    assert speed_json["per_request"][0]["ttfa_s"] == 0.3457

    csv_text = (tmp_path / "results.csv").read_text()
    assert "ttfa_s" in csv_text.splitlines()[0]
    assert "0.3457" in csv_text


def test_omni_send_fn_records_non_stream_ttfa(monkeypatch, tmp_path) -> None:
    wav_bytes = _wav_bytes()
    system_prompt = "Use the unit-test voice style."
    forwarded_args = {}

    async def fake_generate_speech(
        self,
        session,
        api_url,
        model_name,
        sample,
        lang,
        speaker="Ethan",
        max_tokens=None,
        temperature=0.7,
        voice_clone=False,
        stream=False,
        system_prompt=None,
    ):
        forwarded_args["stream"] = stream
        forwarded_args["system_prompt"] = system_prompt
        return wav_bytes, 0.0, {"prompt_tokens": 3, "completion_tokens": 4}

    ticks = iter([10.0, 10.42, 10.43, 10.44])
    monkeypatch.setattr(
        benchmark_omni_seedtts.VoiceCloneOmni,
        "generate_speech",
        fake_generate_speech,
    )
    monkeypatch.setattr(
        benchmark_omni_seedtts.time,
        "perf_counter",
        lambda: next(ticks),
    )

    send_fn = make_send_fn(
        "ming-omni",
        "http://localhost:8000/v1/chat/completions",
        lang="en",
        voice_clone=False,
        speaker="Ethan",
        max_tokens=256,
        temperature=0.7,
        stream=False,
        save_audio_dir=str(tmp_path),
        system_prompt=system_prompt,
    )
    sample = SampleInput(
        sample_id="sample-1",
        ref_text="reference",
        ref_audio="/tmp/ref.wav",
        target_text="hello world",
    )

    result = asyncio.run(send_fn(object(), sample))

    assert result.is_success
    assert forwarded_args == {
        "stream": False,
        "system_prompt": system_prompt,
    }
    assert result.ttfa_s == pytest.approx(0.42)
    assert result.latency_s == pytest.approx(0.44)


def test_streaming_response_records_ttfa(monkeypatch, tmp_path) -> None:
    wav_bytes = _wav_bytes()
    event = {
        "audio": {
            "data": base64.b64encode(wav_bytes).decode("ascii"),
        }
    }
    lines = [
        f"data: {json.dumps(event)}\n".encode(),
        b"data: [DONE]\n",
    ]

    class FakeContent:
        async def iter_any(self):
            for line in lines:
                yield line

    class FakeResponse:
        content = FakeContent()

    ticks = iter([100.25, 101.0])
    monkeypatch.setattr(tts_tasks.time, "perf_counter", lambda: next(ticks))

    result = RequestResult(request_id="stream-1")
    asyncio.run(
        _handle_streaming_response(
            FakeResponse(),
            result,
            100.0,
            str(tmp_path),
        )
    )

    assert result.is_success
    assert result.ttfa_s == pytest.approx(0.25)
    assert result.latency_s == 0.0
    assert result.audio_duration_s == pytest.approx(0.1)
    assert (tmp_path / "stream-1.wav").exists()
