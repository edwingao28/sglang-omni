from __future__ import annotations

import io
import uuid
import wave
from contextlib import asynccontextmanager

import aiohttp
import pytest

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.metrics.performance import build_speed_results
from benchmarks.tasks.tts import make_tts_send_fn


@pytest.mark.asyncio
@pytest.mark.parametrize("with_header", [False, True])
async def test_tts_result_preserves_request_identity_and_wall_clock(with_header):
    wav = io.BytesIO()
    with wave.open(wav, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0" * 480)

    class Response:
        status = 200
        headers = {"X-Request-ID": "speech-worker-id"} if with_header else {}

        async def read(self):
            return wav.getvalue()

    class Session:
        @asynccontextmanager
        async def post(self, url, *, json, headers):
            assert json["input"] == "hello"
            assert uuid.UUID(headers["X-Request-ID"]).version == 4
            yield Response()

    send = make_tts_send_fn("tts", "http://test/speech", no_ref_audio=True)
    result = await send(Session(), SampleInput("sample-1", "", "", "hello"))
    assert result.is_success
    assert result.request_id == "sample-1"
    assert result.server_request_id == ("speech-worker-id" if with_header else None)
    assert 0 < result.request_start_ns <= result.request_end_ns
    row = build_speed_results([result], {}, {})["per_request"][0]
    assert row["server_request_id"] == result.server_request_id
    assert row["request_start_ns"] == result.request_start_ns
    assert row["request_end_ns"] == result.request_end_ns


@pytest.mark.asyncio
async def test_tts_transport_failure_retains_end_timestamp():
    class Session:
        def post(self, *args, **kwargs):
            raise aiohttp.ClientConnectionError("unreachable")

    send = make_tts_send_fn("tts", "http://test/speech", no_ref_audio=True)
    result = await send(Session(), SampleInput("sample-1", "", "", "hello"))
    assert not result.is_success
    assert result.error == "unreachable"
    assert result.server_request_id is None
    assert 0 < result.request_start_ns <= result.request_end_ns
