"""Exercise the edited function bodies on CPU without importing the GPU stack."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import contextlib
import contextvars
import json
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.profiler.event_recorder import emit, get_recorder, host_phase
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

ROOT = Path(__file__).resolve().parents[3]


def load_functions(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    source = (
        next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
        if class_name
        else tree.body
    )
    nodes = [
        n
        for n in source
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 - execute only named functions from this checkout
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace


@pytest.fixture(autouse=True)
def recorder():
    rec = get_recorder()
    rec.stop()
    yield rec
    rec.stop()


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_host_phase_inactive_and_exception_identity(tmp_path, recorder):
    error = RuntimeError("original")
    with pytest.raises(RuntimeError) as caught:
        with host_phase("r", "preprocessing", "service"):
            raise error
    assert caught.value is error
    assert not list(tmp_path.iterdir())
    path = recorder.start("profile", str(tmp_path), "preprocessing")
    with pytest.raises(RuntimeError) as caught:
        with host_phase("r", "preprocessing", "service"):
            raise error
    assert caught.value is error
    rows = events(path)
    assert [r["event_name"] for r in rows] == [
        "host_service_enter",
        "host_service_exit",
    ]
    assert rows[-1]["metadata"]["status"] == "error"
    assert rows[0]["thread_id"] == rows[1]["thread_id"]


@pytest.mark.parametrize("fail", [False, True])
def test_actual_preprocess_worker_is_after_submit_and_preserves_result(
    tmp_path, recorder, fail
):
    rid = contextvars.ContextVar("test_request", default=None)
    value = SimpleNamespace(request_id="r1")
    error = ValueError("handler")

    def compute(payload, **kwargs):
        assert rid.get() == "r1"
        if fail:
            raise error
        return payload

    ns = load_functions(
        "sglang_omni/models/qwen3_tts/request_builders.py",
        ["preprocess_qwen3_tts_payload"],
        {
            "get_recorder": get_recorder,
            "host_phase": host_phase,
            "_HOST_REQUEST_ID": rid,
            "_preprocess_qwen3_tts_payload": compute,
        },
    )
    path = recorder.start("profile", str(tmp_path), "preprocessing")
    scheduler = ThreadedSimpleScheduler(
        ns["preprocess_qwen3_tts_payload"], max_concurrency=1
    )
    thread = threading.Thread(target=scheduler.start)
    thread.start()
    try:
        scheduler.inbox.put(
            IncomingMessage(type="new_request", request_id="r1", data=value)
        )
        result = scheduler.outbox.get(timeout=3)
        assert result.data is (error if fail else value)
        assert result.type == ("error" if fail else "result")
    finally:
        scheduler.stop()
        thread.join(timeout=3)
    assert not thread.is_alive()
    rows = events(path)
    assert [r["event_name"] for r in rows] == [
        "host_handler_submit",
        "host_preprocess_service_enter",
        "host_preprocess_service_exit",
    ]
    assert rows[0]["thread_id"] != rows[1]["thread_id"] == rows[2]["thread_id"]
    assert rows[-1]["metadata"]["status"] == ("error" if fail else "success")
    assert rid.get() is None


@dataclass
class Request:
    stream: bool = False
    extra_params: dict | None = None


def speech_method(encode):
    ns = load_functions(
        "sglang_omni/client/client.py",
        ["speech"],
        {
            "asyncio": asyncio,
            "replace": replace,
            "get_recorder": get_recorder,
            "emit": emit,
            "host_phase": host_phase,
            "encode_audio": encode,
            "SpeechResult": SimpleNamespace,
            "FORMAT_MIME_TYPES": {"wav": "audio/wav"},
            "ClientError": RuntimeError,
        },
        "Client",
    )

    async def generate(*args, **kwargs):
        yield SimpleNamespace(
            audio_data=b"waveform", sample_rate=24000, usage=None, finish_reason="stop"
        )

    return ns["speech"], SimpleNamespace(generate=generate)


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_actual_encode_runs_in_thread_and_closes_error_span(
    tmp_path, recorder, active, fail
):
    error = ValueError("encode")
    calls = []

    def encode(audio, **kwargs):
        calls.append((audio, threading.get_native_id()))
        if fail:
            raise error
        return b"encoded", "audio/wav"

    method, client = speech_method(encode)
    path = recorder.start("profile", str(tmp_path), "coordinator") if active else None
    call = method(client, Request(extra_params={}), request_id="r1")
    if fail:
        with pytest.raises(ValueError) as caught:
            asyncio.run(call)
        assert caught.value is error
    else:
        assert asyncio.run(call).audio_bytes == b"encoded"
    assert calls[0][0] == b"waveform"
    assert calls[0][1] != threading.get_native_id()
    if active:
        rows = events(path)
        assert [r["event_name"] for r in rows] == [
            "host_audio_encode_submit",
            "host_audio_encode_service_enter",
            "host_audio_encode_service_exit",
            "host_audio_encode_resumed",
        ]
        assert rows[1]["thread_id"] == rows[2]["thread_id"] == calls[0][1]
        assert rows[0]["thread_id"] == rows[3]["thread_id"] == threading.get_native_id()
        assert rows[2]["metadata"]["status"] == ("error" if fail else "success")
    else:
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fail", [False, True])
def test_actual_vocoder_batch_has_one_owner_and_all_members(tmp_path, recorder, fail):
    error = ValueError("decode")

    def decode(items):
        if fail:
            raise error
        return [b"a", b"b"], 24000

    ns = load_functions(
        "sglang_omni/models/qwen3_tts/streaming_vocoder.py",
        ["_vocode_payloads", "_vocode_payloads_impl", "_store_vocoder_result"],
        {
            "get_recorder": get_recorder,
            "host_phase": host_phase,
            "threading": threading,
            "time": time,
            "torch": SimpleNamespace(as_tensor=lambda value, **kw: value, long="long"),
            "Qwen3TTSState": SimpleNamespace(
                from_dict=lambda data: SimpleNamespace(audio_codes=data, ref_code_len=0)
            ),
            "audio_waveform_payload": lambda waveform, **kw: {"waveform": waveform},
            "build_usage": lambda state: None,
        },
        "Qwen3TTSStreamingVocoderScheduler",
    )
    cls = type(
        "Vocoder",
        (),
        {
            k: ns[k]
            for k in (
                "_vocode_payloads",
                "_vocode_payloads_impl",
                "_store_vocoder_result",
            )
        },
    )
    obj = cls()
    obj._deterministic_inference = False
    obj._tokenizer = SimpleNamespace(decode=decode)
    payloads = [
        SimpleNamespace(request_id="r1", data=[1]),
        SimpleNamespace(request_id="r2", data=[2]),
    ]
    path = recorder.start("profile", str(tmp_path), "vocoder")
    if fail:
        with pytest.raises(ValueError) as caught:
            asyncio.run(obj._vocode_payloads(payloads))
        assert caught.value is error
    else:
        result = asyncio.run(obj._vocode_payloads(payloads))
        assert all(left is right for left, right in zip(result, payloads))
        assert [p.data for p in payloads] == [{"waveform": b"a"}, {"waveform": b"b"}]
    rows = events(path)
    batch = [r for r in rows if r["event_name"].startswith("host_vocoder_batch")]
    assert len(batch) == 2
    assert {r["request_id"] for r in batch} == {"r1"}
    assert all(r["metadata"]["member_request_ids"] == ["r1", "r2"] for r in batch)
    assert batch[0]["metadata"]["batch_id"] == batch[1]["metadata"]["batch_id"]
    assert batch[1]["metadata"]["status"] == ("error" if fail else "success")


def test_existing_frontend_lifecycle_covers_encode_through_stop(tmp_path, recorder):
    routes = {}

    class Router:
        def post(self, path):
            def register(fn):
                routes[path] = fn
                return fn

            return register

    class Control:
        async def broadcast_start(self, **kwargs):
            assert recorder.is_active()

        async def broadcast_stop(self, **kwargs):
            assert not recorder.is_active()

    ns = load_functions(
        "sglang_omni/serve/launcher.py",
        ["_mount_profiler_routes"],
        {
            "APIRouter": Router,
            "_get_event_recorder": get_recorder,
        },
    )
    ns["_mount_profiler_routes"](
        SimpleNamespace(include_router=lambda router: None), Control(), None
    )

    async def exercise():
        await routes["/start_request_profile"](
            SimpleNamespace(run_id="profile", event_dir=str(tmp_path))
        )
        path = recorder.active_path()
        method, client = speech_method(
            lambda *args, **kwargs: (b"encoded", "audio/wav")
        )
        assert (
            await method(client, Request(extra_params={}), request_id="r1")
        ).audio_bytes == b"encoded"
        await routes["/stop_request_profile"](SimpleNamespace(run_id="profile"))
        return events(path)

    rows = asyncio.run(exercise())
    assert rows[-1]["event_name"] == "host_audio_encode_resumed"
    assert all(row["run_id"] == "profile" for row in rows)


def test_actual_reference_batch_keeps_future_members_and_handoff(tmp_path, recorder):
    rid = contextvars.ContextVar("reference_request", default=None)
    stop = object()
    ns = load_functions(
        "sglang_omni/models/qwen3_tts/request_builders.py",
        ["submit", "_drain", "_run"],
        {
            "concurrent": concurrent,
            "queue": queue,
            "threading": threading,
            "time": time,
            "contextlib": contextlib,
            "get_recorder": get_recorder,
            "host_phase": host_phase,
            "_HOST_REQUEST_ID": rid,
            "_QWEN3_TTS_REF_CODE_BATCH_STOP": stop,
            "torch": SimpleNamespace(inference_mode=contextlib.nullcontext),
        },
        "_Qwen3TTSRefCodeBatcher",
    )
    cls = type("Batcher", (), {name: ns[name] for name in ("submit", "_drain", "_run")})
    batcher = cls()
    batcher._queue = queue.Queue()
    batcher._max_batch_size = 8
    batcher._max_batch_wait_s = 0
    batcher._encode_stream = None
    batcher._speech_tokenizer = SimpleNamespace(
        encode=lambda waveforms, **kw: SimpleNamespace(audio_codes=waveforms)
    )
    path = recorder.start("profile", str(tmp_path), "preprocessing")
    futures = []
    for request_id in ("r1", "r2"):
        token = rid.set(request_id)
        try:
            futures.append(batcher.submit(request_id, 24000))
        finally:
            rid.reset(token)
    batcher._queue.put(stop)
    handoffs = []

    def handoff(outcomes):
        assert not any(future.done() for future in futures)
        handoffs.append(sorted(outcomes))

    batcher._synchronize_outcomes = handoff
    batcher._run()
    assert handoffs == [[0, 1]]
    assert [future.result() for future in futures] == ["r1", "r2"]
    rows = events(path)
    assert [row["event_name"] for row in rows] == [
        "host_reference_batch_enter",
        "host_reference_handoff_sync_enter",
        "host_reference_handoff_sync_exit",
        "host_reference_batch_exit",
    ]
    assert {row["request_id"] for row in rows} == {"r1"}
    assert all(row["metadata"]["member_request_ids"] == ["r1", "r2"] for row in rows)
    assert len({row["metadata"]["batch_id"] for row in rows}) == 1


@pytest.mark.parametrize("has_codes", [False, True])
def test_actual_output_adapter_keeps_values_and_conditional_cpu_phase(
    tmp_path, recorder, has_codes
):
    import torch

    ns = load_functions(
        "sglang_omni/models/qwen3_tts/request_builders.py",
        ["apply_sglang_qwen3_tts_result", "_apply_sglang_qwen3_tts_result"],
        {
            "host_phase": host_phase,
            "torch": torch,
            "StagePayload": SimpleNamespace,
            "time": time,
            "_qwen3_tts_finish_reason": lambda data: "stop",
        },
    )
    payload = SimpleNamespace(request_id="r1", request=object())
    data = SimpleNamespace(
        ref_code=None,
        ref_code_len=0,
        output_codes=[torch.tensor([1, 2])] if has_codes else [],
        engine_start_s=time.perf_counter(),
    )
    path = recorder.start("profile", str(tmp_path), "tts_engine")
    result = ns["apply_sglang_qwen3_tts_result"](payload, data)
    assert result.request is payload.request
    assert result.request_id == "r1"
    assert result.data["audio_codes"].tolist() == ([[1, 2]] if has_codes else [])
    assert result.data["completion_tokens"] == int(has_codes)
    names = [r["event_name"] for r in events(path)]
    assert ("host_output_codes_cpu_enter" in names) is has_codes
    assert (
        names[0] == "host_result_adapter_enter"
        and names[-1] == "host_result_adapter_exit"
    )
