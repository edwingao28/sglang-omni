# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Ming V1 streaming TTS client/API compatibility."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
import types
from typing import Any

import numpy as np
import pytest

sys.modules.setdefault("msgpack", types.SimpleNamespace(packb=None, unpackb=None))
zmq_asyncio_stub = types.SimpleNamespace(Context=object, Socket=object)
sys.modules.setdefault(
    "zmq",
    types.SimpleNamespace(
        PUSH=1,
        PULL=2,
        PUB=3,
        SUB=4,
        SUBSCRIBE=5,
        asyncio=zmq_asyncio_stub,
    ),
)
sys.modules.setdefault("zmq.asyncio", zmq_asyncio_stub)
sys.modules.setdefault("uvicorn", types.SimpleNamespace(run=None))

from sglang_omni.client import (
    Client,
    CompletionStreamChunk,
    GenerateChunk,
    GenerateRequest,
)
from sglang_omni.proto import StreamMessage
from sglang_omni.serve.openai_api import _chat_stream
from sglang_omni.serve.protocol import ChatCompletionRequest


def _sse_payload(event: str) -> dict[str, Any]:
    assert event.startswith("data: ")
    return json.loads(event[len("data: ") :])


def test_set_audio_data_keeps_numpy_audio_data_without_truthiness_fallback() -> None:
    chunk = GenerateChunk(request_id="req-1")
    audio_data = np.array([], dtype=np.float32)
    fallback_audio = np.array([0.5], dtype=np.float32)

    Client._set_audio_data(
        chunk,
        {
            "audio_data": audio_data,
            "audio": fallback_audio,
            "sample_rate": 16000,
        },
    )

    assert chunk.audio_data is audio_data
    assert chunk.sample_rate == 16000
    assert chunk.modality == "audio"


@pytest.mark.asyncio
async def test_completion_stream_uses_chunk_sample_rate_and_skips_empty_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_audio_to_base64(
        audio: Any,
        *,
        sample_rate: int,
        output_format: str,
    ) -> str:
        calls.append(
            {
                "audio": audio,
                "sample_rate": sample_rate,
                "output_format": output_format,
            }
        )
        return f"encoded-{sample_rate}-{output_format}"

    monkeypatch.setattr(
        "sglang_omni.client.client.audio_to_base64",
        fake_audio_to_base64,
    )

    class StreamingClient(Client):
        async def generate(self, request: GenerateRequest, request_id: str | None = None):
            yield GenerateChunk(request_id=request_id or "req-1", text="hello")
            yield GenerateChunk(
                request_id=request_id or "req-1",
                modality="audio",
                audio_data=np.array([], dtype=np.float32),
                sample_rate=16000,
            )
            yield GenerateChunk(
                request_id=request_id or "req-1",
                modality="audio",
                audio_data=np.array([0.1, -0.1], dtype=np.float32),
                sample_rate=22050,
            )
            yield GenerateChunk(request_id=request_id or "req-1", finish_reason="stop")

    client = StreamingClient(coordinator=None)  # type: ignore[arg-type]

    chunks = [
        chunk
        async for chunk in client.completion_stream(
            GenerateRequest(prompt="hi"),
            request_id="req-1",
            audio_format="wav",
        )
    ]

    assert [(chunk.text, chunk.modality, chunk.audio_b64, chunk.finish_reason) for chunk in chunks] == [
        ("hello", "text", None, None),
        ("", "audio", "encoded-22050-wav", None),
        ("", "text", None, "stop"),
    ]
    assert len(calls) == 1
    assert calls[0]["sample_rate"] == 22050


@pytest.mark.asyncio
async def test_completion_real_wav_uses_chunk_sample_rate() -> None:
    class NonStreamingClient(Client):
        async def generate(self, request: GenerateRequest, request_id: str | None = None):
            del request
            yield GenerateChunk(
                request_id=request_id or "req-1",
                modality="audio",
                audio_data=np.array([0.1, -0.1], dtype=np.float32),
                sample_rate=16000,
                finish_reason="stop",
            )

    client = NonStreamingClient(coordinator=None)  # type: ignore[arg-type]

    result = await client.completion(
        GenerateRequest(prompt="hi", stream=False),
        request_id="req-1",
        audio_format="wav",
    )
    assert result.audio is not None
    wav = base64.b64decode(result.audio.data)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[24:28])[0] == 16000


@pytest.mark.asyncio
async def test_completion_stream_real_wav_uses_chunk_sample_rate() -> None:
    class StreamingClient(Client):
        async def generate(self, request: GenerateRequest, request_id: str | None = None):
            del request
            yield GenerateChunk(
                request_id=request_id or "req-1",
                modality="audio",
                audio_data=np.array([0.1, -0.1], dtype=np.float32),
                sample_rate=16000,
            )

    client = StreamingClient(coordinator=None)  # type: ignore[arg-type]

    [chunk] = [
        chunk
        async for chunk in client.completion_stream(
            GenerateRequest(prompt="hi"),
            request_id="req-1",
            audio_format="wav",
        )
    ]
    wav = base64.b64decode(chunk.audio_b64 or "")

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[24:28])[0] == 16000


def test_default_stream_builder_propagates_streaming_telemetry_fields() -> None:
    chunk = Client._default_stream_builder(
        "req-1",
        StreamMessage(
            request_id="req-1",
            from_stage="talker_stream",
            stage_name="talker_stream",
            modality="audio",
            chunk={
                "audio_data": np.array([0.1], dtype=np.float32),
                "sample_rate": 16000,
                "talker_queue_depth": 3,
                "segmenter_first_emit_ms": 12.5,
                "stage_times_ms": {"segmenter_first_emit": 12.5},
            },
        ),
    )

    assert chunk.sample_rate == 16000
    assert chunk.talker_queue_depth == 3
    assert chunk.segmenter_first_emit_ms == 12.5
    assert chunk.stage_times_ms == {"segmenter_first_emit": 12.5}


def test_default_result_builder_treats_talker_stream_terminal_as_audio() -> None:
    chunk = Client._default_result_builder(
        "req-1",
        {
            "decode": {"text": "hello"},
            "talker_stream": {
                "audio_data": np.array([0.1, -0.1], dtype=np.float32),
                "sample_rate": 16000,
            },
        },
    )

    assert chunk.text == "hello"
    assert chunk.modality == "audio"
    assert chunk.sample_rate == 16000
    assert np.asarray(chunk.audio_data).tolist() == pytest.approx([0.1, -0.1])


@pytest.mark.asyncio
async def test_chat_stream_includes_telemetry_and_payload_finish_chunk() -> None:
    class StreamingChatClient:
        async def completion_stream(
            self,
            request: GenerateRequest,
            *,
            request_id: str,
            audio_format: str,
        ):
            del request, audio_format
            chunk = CompletionStreamChunk(
                request_id=request_id,
                text="hello",
                modality="text",
                finish_reason="stop",
                stage_name="decode",
            )
            chunk.sample_rate = 16000
            chunk.talker_queue_depth = 2
            chunk.stage_times_ms = {"segmenter_first_emit": 11.0}
            yield chunk

    events = [
        event
        async for event in _chat_stream(
            StreamingChatClient(),  # type: ignore[arg-type]
            GenerateRequest(prompt="hi"),
            "req-1",
            "chatcmpl-req-1",
            123,
            "ming",
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                modalities=["text", "audio"],
            ),
            "wav",
        )
    ]

    payload = _sse_payload(events[0])
    assert payload["choices"][0]["delta"]["content"] == "hello"
    assert "stage_name" not in payload
    assert "sample_rate" not in payload
    assert "talker_queue_depth" not in payload
    assert "stage_times_ms" not in payload
    telemetry = payload["choices"][0]["delta"]["sglang_omni"]
    assert telemetry["stage_name"] == "decode"
    assert telemetry["sample_rate"] == 16000
    assert telemetry["talker_queue_depth"] == 2
    assert telemetry["stage_times_ms"]["segmenter_first_emit"] == 11.0
    assert "thinker_first_text" in telemetry["stage_times_ms"]
    assert "segmenter_first_emit_ms" not in telemetry["stage_times_ms"]
    assert "first_text_ms" not in telemetry["stage_times_ms"]

    finish_payload = _sse_payload(events[1])
    assert finish_payload["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_stream_emits_text_from_audio_modality_finish_chunk() -> None:
    class StreamingChatClient:
        async def completion_stream(
            self,
            request: GenerateRequest,
            *,
            request_id: str,
            audio_format: str,
        ):
            del request, audio_format
            yield CompletionStreamChunk(
                request_id=request_id,
                text="terminal text",
                modality="audio",
                audio_b64=None,
                finish_reason="stop",
                stage_name="talker_stream",
            )

    events = [
        event
        async for event in _chat_stream(
            StreamingChatClient(),  # type: ignore[arg-type]
            GenerateRequest(prompt="hi"),
            "req-1",
            "chatcmpl-req-1",
            123,
            "ming",
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                modalities=["text", "audio"],
            ),
            "wav",
        )
    ]

    payload = _sse_payload(events[0])
    assert payload["choices"][0]["delta"]["content"] == "terminal text"
    assert payload["choices"][0]["finish_reason"] is None

    finish_payload = _sse_payload(events[1])
    assert finish_payload["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_stream_aborts_request_on_cancellation() -> None:
    class CancellableClient:
        def __init__(self) -> None:
            self.aborted: list[str] = []

        async def completion_stream(
            self,
            request: GenerateRequest,
            *,
            request_id: str,
            audio_format: str,
        ):
            del request, audio_format
            yield CompletionStreamChunk(request_id=request_id, text="first")
            await asyncio.Event().wait()

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    client = CancellableClient()
    stream = _chat_stream(
        client,  # type: ignore[arg-type]
        GenerateRequest(prompt="hi"),
        "req-cancel",
        "chatcmpl-req-cancel",
        123,
        "ming",
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        ),
        "wav",
    )

    first = await stream.__anext__()
    assert _sse_payload(first)["choices"][0]["delta"]["content"] == "first"

    pending = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert client.aborted == ["req-cancel"]


@pytest.mark.asyncio
async def test_chat_stream_aborts_request_on_generator_close() -> None:
    class ClosableClient:
        def __init__(self) -> None:
            self.aborted: list[str] = []

        async def completion_stream(
            self,
            request: GenerateRequest,
            *,
            request_id: str,
            audio_format: str,
        ):
            del request, audio_format
            yield CompletionStreamChunk(request_id=request_id, text="first")
            await asyncio.Event().wait()

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    client = ClosableClient()
    stream = _chat_stream(
        client,  # type: ignore[arg-type]
        GenerateRequest(prompt="hi"),
        "req-close",
        "chatcmpl-req-close",
        123,
        "ming",
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        ),
        "wav",
    )

    await stream.__anext__()
    await stream.aclose()

    assert client.aborted == ["req-close"]
