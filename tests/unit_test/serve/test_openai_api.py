# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import ClientError, GenerateChunk
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _speech_stream


class FailingSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        yield GenerateChunk(
            request_id="speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
        )
        raise ClientError("stream failed")


def test_speech_stream_returns_error_event_after_chunk_failure() -> None:
    """Preserves deterministic SSE termination after a mid-stream client error."""
    client = TestClient(create_app(FailingSpeechClient(), model_name="s2-pro"))

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "model": "s2-pro",
            "input": "hello",
            "stream": True,
            "response_format": "wav",
        },
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        events = []
        done = False
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                done = True
                break
            events.append(json.loads(payload))

    assert done
    assert len(events) == 2
    assert events[0]["audio"] is not None
    assert events[0]["finish_reason"] is None
    assert events[1]["audio"] is None
    assert events[1]["finish_reason"] == "error"
    assert events[1]["error"] == {
        "type": "ClientError",
        "message": "stream failed",
    }


@pytest.mark.asyncio
async def test_speech_stream_skips_empty_nonterminal_audio_chunks() -> None:
    class EmptyThenAudioClient:
        async def generate(self, request: Any, request_id: str | None = None):
            del request
            yield GenerateChunk(
                request_id=request_id or "speech-1",
                modality="audio",
                audio_data=np.array([], dtype=np.float32),
                sample_rate=24000,
            )
            yield GenerateChunk(
                request_id=request_id or "speech-1",
                modality="audio",
                audio_data=np.array([0.0, 0.1], dtype=np.float32),
                sample_rate=24000,
            )

    events = [
        event
        async for event in _speech_stream(
            EmptyThenAudioClient(),  # type: ignore[arg-type]
            gen_req=None,  # type: ignore[arg-type]
            request_id="speech-1",
            response_format="wav",
            speed=1.0,
        )
    ]

    payloads = [
        json.loads(event[len("data: ") :])
        for event in events
        if event != "data: [DONE]\n\n"
    ]

    assert len(payloads) == 2
    assert payloads[0]["audio"] is not None
    assert payloads[0]["index"] == 0
    assert payloads[1]["audio"] is None
    assert payloads[1]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_speech_stream_aborts_request_on_cancellation() -> None:
    class CancellableSpeechClient:
        def __init__(self) -> None:
            self.aborted: list[str] = []

        async def generate(self, request: Any, request_id: str | None = None):
            del request
            yield GenerateChunk(
                request_id=request_id or "speech-1",
                modality="audio",
                audio_data=np.array([0.0, 0.1], dtype=np.float32),
                sample_rate=24000,
            )
            await asyncio.Event().wait()

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    client = CancellableSpeechClient()
    stream = _speech_stream(
        client,  # type: ignore[arg-type]
        gen_req=None,  # type: ignore[arg-type]
        request_id="speech-cancel",
        response_format="wav",
        speed=1.0,
    )

    first = await stream.__anext__()
    assert json.loads(first[len("data: ") :])["audio"] is not None

    pending = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert client.aborted == ["speech-cancel"]


@pytest.mark.asyncio
async def test_speech_stream_aborts_request_on_generator_close() -> None:
    class ClosableSpeechClient:
        def __init__(self) -> None:
            self.aborted: list[str] = []

        async def generate(self, request: Any, request_id: str | None = None):
            del request
            yield GenerateChunk(
                request_id=request_id or "speech-1",
                modality="audio",
                audio_data=np.array([0.0, 0.1], dtype=np.float32),
                sample_rate=24000,
            )
            await asyncio.Event().wait()

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    client = ClosableSpeechClient()
    stream = _speech_stream(
        client,  # type: ignore[arg-type]
        gen_req=None,  # type: ignore[arg-type]
        request_id="speech-close",
        response_format="wav",
        speed=1.0,
    )

    first = await stream.__anext__()
    assert json.loads(first[len("data: ") :])["audio"] is not None

    await stream.aclose()

    assert client.aborted == ["speech-close"]
